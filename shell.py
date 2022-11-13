from p_sharp import Main

while True:
    try:
        input_ = input("p# > ")
        result, error = Main(input_, '<Shell>')
        if error: print(error.as_string())
        else: 
            for res in result: print(res)
    except KeyboardInterrupt:
        exit()