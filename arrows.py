def add_arrows(str, start, end=None):
    if start > len(str):
        return str
       
    if end:
        if end > len(str):
            return str
        end = end-start+1
    else: end = 1
    if end == 0: end = 1

    string = str + '\n'
    string += ' '*start + '^'*end
    return string