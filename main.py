from p_sharp import Main
from sys import argv
from os import path

try:
    if len(argv) == 1:
        ShellNum = 0

        while True:
            input_ = input("p# > ")
            result, error = Main(input_, f'<Shell#{ShellNum}>')
            if error: print(error.as_string())
            else: 
                for res in result: print(res)
            ShellNum += 1
    else:
        if not path.isfile(argv[1]):
            print(f"Can't open file '{argv[1]}': No such file or directory")

        with open(argv[1], 'r') as f:
            result, error = Main(f.read(), path.split(f.name)[1])
            if error: print(error.as_string())
except KeyboardInterrupt:
    exit()
    