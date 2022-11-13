from p_sharp import Main

with open('aaaa.ps', 'r') as f:
    result, error = Main(f.read(), f.name)
    if error: print(error.as_string())
    else: print(result)