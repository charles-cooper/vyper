from collections import defaultdict

from vyper.exceptions import ExceptionList, InitializerException
from vyper.semantics.analysis.base import InitializesInfo, UsesInfo
from pathlib import Path
from vyper.exceptions import StructureException
from vyper.semantics.types.module import ModuleT


def validate_compilation_target(module_t: ModuleT):
    _validate_global_initializes_constraint(module_t)
    _validate_abstract_resolutions(module_t)


def _hook_qualifier(module_t: ModuleT, hook_fn) -> str:
    # try to reconstruct a friendly name like "base.foo"
    hook_module_t = hook_fn.decl_node.module_node._metadata["type"]
    modinfo = module_t.find_module_info(hook_module_t)
    if modinfo is not None:
        alias = modinfo.alias
    else:
        # fallback to filename stem
        try:
            p = hook_fn.decl_node.module_node.path  # type: ignore[attr-defined]
            alias = Path(p).stem
        except Exception:
            alias = str(hook_module_t)
    return f"{alias}.{hook_fn.name}"


def _validate_abstract_resolutions(module_t: ModuleT):
    # Collect providers from direct imports
    providers = {}
    for mi in module_t.imported_modules.values():
        provs = mi.module_t.decl_node._metadata.get("provides_map", [])
        for hook_fn, impl_fn, hook_str, node in provs:
            providers.setdefault(hook_fn, []).append((mi, impl_fn))

    # Collect obligations from direct imports
    obligations = set()
    for mi in module_t.imported_modules.values():
        for hook_fn in mi.module_t.decl_node._metadata.get("abstract_obligations", set()):
            obligations.add(hook_fn)

    # Collect resolutions from this module
    resolutions = {}
    for entry in module_t.decl_node._metadata.get("resolutions_map", []):
        resolutions.setdefault(entry["hook"], []).append(entry)

    # 1) Diamonds: if multiple providers exist for a hook, require a resolution
    for hook_fn, provs in providers.items():
        if len(provs) > 1 and hook_fn not in resolutions:
            hook_q = _hook_qualifier(module_t, hook_fn)
            raise StructureException(f"Multiple providers for {hook_q}; add a resolution", module_t.decl_node)

    # 2) Validate each resolution entry
    for hook_fn, entries in resolutions.items():
        provs = providers.get(hook_fn, [])
        prov_impls = {impl for (_mi, impl) in provs}

        for e in entries:
            mode = e["mode"]
            impl = e["impl"]
            node = e["node"]

            # signature check: args and return type must match
            def _sig_ok(a, b):
                if len(a.argument_types) != len(b.argument_types):
                    return False
                for x, y in zip(a.argument_types, b.argument_types):
                    if not x.compare_type(y):
                        return False
                if (a.return_type is None) != (b.return_type is None):
                    return False
                if a.return_type is not None and not a.return_type.compare_type(b.return_type):
                    return False
                return True

            if not _sig_ok(impl, hook_fn):
                hook_q = _hook_qualifier(module_t, hook_fn)
                raise StructureException(f"Implementation signature mismatch for {hook_q}", node)

            if mode == "accept":
                if impl not in prov_impls:
                    hook_q = _hook_qualifier(module_t, hook_fn)
                    raise StructureException(f"No provider found for {hook_q}", node)
            elif mode == "override":
                # ensure replaces list refers to actual providers
                for r in e["replaces"]:
                    if r not in prov_impls:
                        hook_q = _hook_qualifier(module_t, hook_fn)
                        raise StructureException(f"Invalid override for {hook_q}", node)
            else:
                # fresh mapping, allowed
                pass

    # 3) Unresolved obligations: for each required hook, ensure there is a resolution
    for hook_fn in obligations:
        if hook_fn not in resolutions:
            hook_q = _hook_qualifier(module_t, hook_fn)
            raise StructureException(f"Unresolved abstract method {hook_q}", module_t.decl_node)


def _collect_used_modules_r(module_t):
    ret: defaultdict[ModuleT, list[UsesInfo]] = defaultdict(list)

    for uses_decl in module_t.uses_decls:
        for used_module in uses_decl._metadata["uses_info"].used_modules:
            ret[used_module.module_t].append(uses_decl)

            # recurse
            used_modules = _collect_used_modules_r(used_module.module_t)
            for k, v in used_modules.items():
                ret[k].extend(v)

    # also recurse into modules used by initialized modules
    for i in module_t.initialized_modules:
        used_modules = _collect_used_modules_r(i.module_info.module_t)
        for k, v in used_modules.items():
            ret[k].extend(v)

    return ret


def _collect_initialized_modules_r(module_t, seen=None):
    seen: dict[ModuleT, InitializesInfo] = seen or {}

    # list of InitializedInfo
    initialized_infos = module_t.initialized_modules

    for i in initialized_infos:
        initialized_module_t = i.module_info.module_t
        if initialized_module_t in seen:
            seen_nodes = (i.node, seen[initialized_module_t].node)
            raise InitializerException(f"`{i.module_info.alias}` initialized twice!", *seen_nodes)
        seen[initialized_module_t] = i

        _collect_initialized_modules_r(initialized_module_t, seen)

    return seen


# validate that each module which is `used` in the import graph is
# `initialized`.
def _validate_global_initializes_constraint(module_t: ModuleT):
    all_used_modules = _collect_used_modules_r(module_t)
    all_initialized_modules = _collect_initialized_modules_r(module_t)

    err_list = ExceptionList()

    for u, uses in all_used_modules.items():
        if u not in all_initialized_modules:
            msg = f"module `{u}` is used but never initialized!"

            # construct a hint if the module is in scope
            hint = None
            found_module = module_t.find_module_info(u)
            if found_module is not None:
                # TODO: do something about these constants
                if str(module_t) in ("<unknown>", "VyperContract.vy"):
                    module_str = "the top level of your main contract"
                else:
                    module_str = f"`{module_t}`"
                hint = f"add `initializes: {found_module.alias}` to {module_str}"

            err_list.append(InitializerException(msg, *uses, hint=hint))

    err_list.raise_if_not_empty()
