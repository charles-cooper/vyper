in_ = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
out = "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"

d = dict(zip(in_,out))

def rot13(s):
    return "".join(d.get(c,c) for c in s)

