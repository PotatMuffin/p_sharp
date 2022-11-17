from __future__ import annotations
from arrows import add_arrows
import math
import string

#############################
# constants
#############################

DIGITS = string.digits
LETTERS = f"{string.ascii_letters}_"
characters = f"{string.ascii_letters}{string.digits}{string.punctuation} \t"

#############################
# Errors
#############################

class Error:
    def __init__(self, fn, line:str, error_name:str, details:str, pos_start, pos_end=None) -> None:
        self.fn = fn
        self.line = line
        self.error = error_name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

    def as_string(self):
        return f"{self.error}: '{self.details}'\nFile: {self.fn}\n\n{add_arrows(self.line, self.pos_start, self.pos_end)}"

class IllegalCharError(Error):
    def __init__(self, fn, line: str, details: str, pos_start, pos_end=None) -> None:
        super().__init__(fn, line, "Illegal Character", details, pos_start, pos_end)

class InvalidSyntaxError(Error):
    def __init__(self, fn, line: str, details: str, pos_start, pos_end=None) -> None:
        super().__init__(fn, line, 'Invalid Syntax', details, pos_start, pos_end)
    
class ZeroDivisionError(Error):
    def __init__(self, fn, line: str, details: str, pos_start, pos_end=None) -> None:
        super().__init__(fn, line, 'Zero Division Error', details, pos_start, pos_end)

class ValueError(Error):
    def __init__(self, fn, line: str, details: str, pos_start, pos_end=None) -> None:
        super().__init__(fn, line, 'Value Error', details, pos_start, pos_end)

class NameError(Error):
    def __init__(self, fn, line: str, details: str, pos_start, pos_end=None) -> None:
        super().__init__(fn, line, 'Name Error', details, pos_start, pos_end)

class TypeError(Error):
    def __init__(self, fn, line: str, details: str, pos_start, pos_end=None) -> None:
        super().__init__(fn, line, 'Type Error', details, pos_start, pos_end)

class IndexError(Error):
    def __init__(self, fn, line: str, details: str, pos_start, pos_end=None) -> None:
        super().__init__(fn, line, "Index Error", details, pos_start, pos_end)

#############################
#Symbol Table
#############################

class SymbolTable:
    def __init__(self, parent:SymbolTable=None) -> None:
        self.symbols = {}
        self.keywords = []
        self.parent:SymbolTable = parent
    
    def get_variable(self, variable):
        value = self.symbols.get(variable, None)
        if self.parent and not value:
            value = self.parent.symbols.get(variable, None)
        return value
    
    def add_keywords(self, *kw):
        for keyword in kw:
            self.keywords.append(keyword)

    def assign(self, var_name, value):
        self.symbols[var_name] = value
    
    def assign_multiple(self, *vars:tuple[tuple]):
        for var in vars:
            self.symbols[var[0]] = var[1]

#############################
# Position
#############################

class Position:
    def __init__(self, idx:int, ln:int, text:str, fn:str) -> None:
        self.idx = idx
        self.col = idx
        self.ln = ln
        self.text = text
        self.fn = fn
        self.current_char = self.text[self.idx] if self.idx < len(self.text) else None
        self.set_current_line()

    def set_current_line(self):
        split_text = self.text.split('\n')
        if self.ln < len(split_text):
            self.current_line = split_text[self.ln-1]
        else:
            self.current_line = split_text[-1]
        return self.current_line
    
    def advance(self):
        self.col += 1
        self.idx += 1
        self.current_char = self.text[self.col] if self.col < len(self.text) else None
        if self.current_char == '\n':
            self.ln += 1
            self.idx = -1
            self.set_current_line()

    def back(self):
        if self.current_char == '\n':
            self.ln -= 1
            self.idx = len(self.set_current_line())

        self.col -= 1
        self.idx -= 1
        self.current_char = self.text[self.col] if self.col < len(self.text) else None
    
    def back_line(self):
        self.ln -= 1
        self.idx = len(self.set_current_line())-1

    def copy(self):
        return Position(self.idx, self.ln, self.text, self.fn)

#############################
# Token
#############################

TT_PLUS = "PLUS"
TT_MIN = "MIN"
TT_MUL = "MUL"
TT_DIV = "DIV"
TT_POW = 'POW'

TT_COMMA = 'COMMA'

TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_LSQUARE = "LSQUARE"
TT_RSQUARE = "RSQUARE"

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_STR = 'STR'
TT_FUNC = "FUNC"

TT_IDENTIFIER = "IDENTIFIER"
TT_KEYWORD = "KEYWORD"

TT_EQ = "EQ"
TT_IS = "IS"
TT_ISNOT = "ISNOT"
TT_GT = "GT" # greater than
TT_GTE = "GTE" # greater than or equals
TT_LT = "LT" # lower than
TT_LTE = "LTE" # lower than or equals

TT_NEWLINE = "NEWLINE"
TT_EOF = "EOF"

class Token:
    def __init__(self, type, value=None, pos_start:Position=None, pos_end:Position=None) -> None:
        self.type = type
        self.value = value
        self.pos_start = pos_start.copy() if pos_start else None
        self.pos_end = pos_end.copy() if pos_end else None

        if not self.pos_end and self.pos_start:
            self.pos_end = pos_start.copy()
    
    def __repr__(self) -> str:
        return f"{self.type}" if self.value != 0 and not self.value else f"{self.type}:{self.value}" 

    def matches(self, type=None, value=None, token:Token=None):
        if token: return self.matches(token.type, token.value)
        return self.type == type and self.value == value

#############################
# Lexer
#############################

class Lexer:
    def __init__(self, input, fn) -> None:
        self.pos = Position(0, 1, input, fn)
    
    def comment(self):
        while self.pos.current_char:
            if self.pos.current_char in ';\n':
                break
            self.pos.advance()
    
    def make_greater_than(self):
        pos = self.pos.copy()
        self.pos.advance()
        if self.pos.current_char == '=':
            return Token(type=TT_GTE, pos_start=pos, pos_end=self.pos)
        self.pos.back()
        return Token(type=TT_GT, pos_start=pos)
    
    def make_lower_than(self):
        pos = self.pos.copy()
        self.pos.advance()
        if self.pos.current_char == '=':
            return Token(type=TT_LTE, pos_start=pos, pos_end=self.pos.copy())
        self.pos.back()
        return Token(type=TT_LT, pos_start=pos)
    
    def make_equals(self):
        pos = self.pos.copy()
        self.pos.advance()
        if self.pos.current_char == '=':
            return Token(type=TT_IS, pos_start=pos, pos_end=self.pos.copy())
        self.pos.back()
        return Token(type=TT_EQ, pos_start=pos)
    
    def make_not_equals(self):
        pos = self.pos.copy()
        self.pos.advance()
        if self.pos.current_char == '=':
            return Token(type=TT_ISNOT, pos_start=pos, pos_end=self.pos.copy())
        self.pos.back()
        return Token(type=TT_KEYWORD, value='not', pos_start=pos)

    def make_identifier(self):
        pos = self.pos.copy()
        identifier = ''

        while self.pos.current_char and self.pos.current_char in LETTERS + DIGITS:
            identifier += self.pos.current_char
            self.pos.advance()
        
        pos_end = self.pos.copy()
        pos_end.back()
        
        return Token(type=TT_IDENTIFIER, value=identifier, pos_start=pos, pos_end=pos_end) if identifier not in GlobalSymbolTable.keywords else Token(type=TT_KEYWORD, value=identifier, pos_start=pos, pos_end=pos_end)

    def make_digit(self):
        pos = self.pos.copy()
        digit = ''
        dot_count = 0

        while self.pos.current_char and self.pos.current_char in DIGITS + '.':
            if self.pos.current_char == '.':
                if dot_count == 1:
                    break
                dot_count += 1
            
            digit += self.pos.current_char
            self.pos.advance()

        return Token(type=TT_INT, value=digit, pos_start=pos, pos_end=self.pos.copy()) if dot_count == 0 else Token(type=TT_FLOAT, value=digit, pos_start=pos, pos_end=self.pos.copy())

    def make_string(self):
        pos = self.pos.copy()
        string = ''

        self.pos.advance()
        while self.pos.current_char and self.pos.current_char in characters:
            if self.pos.current_char in '"\'': 
                self.pos.advance()
                break

            string += self.pos.current_char
            self.pos.advance()
        
        pos_end = self.pos.copy()
        pos_end.back()

        return Token(type=TT_STR, value=string, pos_start=pos, pos_end=pos_end)
    
    def make_tokens(self):
        tokens = []
        
        chars = {
        '+': lambda: tokens.append(Token(type=TT_PLUS, pos_start=self.pos.copy())), 
        '-': lambda: tokens.append(Token(type=TT_MIN, pos_start=self.pos.copy())), 
        '*': lambda: tokens.append(Token(type=TT_MUL, pos_start=self.pos.copy())), 
        '/': lambda: tokens.append(Token(type=TT_DIV, pos_start=self.pos.copy())),
        '(': lambda: tokens.append(Token(type=TT_LPAREN, pos_start=self.pos.copy())),
        ')': lambda: tokens.append(Token(type=TT_RPAREN, pos_start=self.pos.copy())),
        '[': lambda: tokens.append(Token(type=TT_LSQUARE, pos_start=self.pos.copy())),
        ']': lambda: tokens.append(Token(type=TT_RSQUARE, pos_start=self.pos.copy())),
        '^': lambda: tokens.append(Token(type=TT_POW, pos_start=self.pos.copy())),
        ';': lambda: tokens.append(Token(type=TT_NEWLINE, pos_start=self.pos.copy())),
        ',': lambda: tokens.append(Token(type=TT_COMMA, pos_start=self.pos.copy())),
        '!': lambda: tokens.append(self.make_not_equals()),
        '=': lambda: tokens.append(self.make_equals()),
        '>': lambda: tokens.append(self.make_greater_than()),
        '<': lambda: tokens.append(self.make_lower_than())
        }

        while self.pos.current_char:
            if self.pos.current_char == '#':
                self.comment()
            if self.pos.current_char in ' \t':
                self.pos.advance()
                continue
            if self.pos.current_char in DIGITS:
                tokens.append(self.make_digit())
                continue
            if self.pos.current_char in LETTERS:
                tokens.append(self.make_identifier())
                continue
            if self.pos.current_char in '"\'':
                tokens.append(self.make_string())
                continue
            if self.pos.current_char == '\n':
                pos = self.pos.copy()
                pos.back_line()
                tokens.append(Token(type=TT_NEWLINE, pos_start=pos))
                self.pos.advance()
                continue
            if self.pos.current_char not in chars:
                return [], IllegalCharError(self.pos.fn, self.pos.current_line, self.pos.current_char, self.pos.idx)

            chars[self.pos.current_char]()
            self.pos.advance()

        tokens.append(Token(type=TT_EOF, pos_start=self.pos))
        return tokens, None

#############################
# Nodes
#############################

class BinOpNode:
    def __init__(self, l:Token, op:Token, r:Token) -> None:
        self.left = l
        self.op = op
        self.right = r

    def __repr__(self) -> str:
        return f"({self.left} {self.op} {self.right})"
    
class UnaryOpNode:
    def __init__(self, op_token:Token, node:Token) -> None:
        self.op_token = op_token
        self.node = node

    def __repr__(self) -> str:
        return f"({self.op_token} {self.node})"

class NumNode:
    def __init__(self, tok:Token) -> None:
        self.token = tok
    
    def __repr__(self) -> str:
        return f"{self.token}"

class StrNode:
    def __init__(self, tok:Token) -> None:
        self.token = tok
    
    def __repr__(self) -> str:
        return f"{self.token}"

class VarAssignNode:
    def __init__(self, var_name:Token, value) -> None:
        self.var_name = var_name
        self.value = value

    def __repr__(self) -> str:
        return f"({self.var_name} = {self.value})"

class VarNode:
    def __init__(self, var:Token) -> None:
        self.var = var
    
    def __repr__(self) -> str:
        return f"{self.var}"

class IndexAccessNode:
    def __init__(self, token:Token, index:NumNode) -> None:
        self.token = token
        self.index = index

    def __repr__(self) -> str:
        return f"{self.token}[{self.index}]"

class IfNode:
    def __init__(self, condition, expressions:ListNode, else_:ListNode=None) -> None:
        self.condition = condition
        self.exprs = expressions
        self.else_ = else_

    def __repr__(self) -> str:
        return f"if {self.condition} then {self.exprs}"

class WhileNode:
    def __init__(self, condition, expressions:ListNode) -> None:
        self.condition = condition
        self.exprs = expressions
    
    def __repr__(self) -> str:
        return f"while {self.condition} then {self.exprs}"

class ForNode:
    def __init__(self, var_name, expr, expressions:list) -> None:
        self.var_name = var_name
        self.expr = expr
        self.exprs = expressions
    
    def __repr__(self) -> str:
        return f"for {self.var_name} in {self.expr} then {self.exprs}"

class FuncDefNode:
    def __init__(self, func_name:Token, arguments:list[Token], expressions:list, token) -> None:
        self.func_name = func_name
        self.args = arguments
        self.exprs = expressions
        self.token = token
    
    def __repr__(self) -> str:
        return f"def {self.func_name}({self.args}) then {self.exprs}"

class CallNode:
    def __init__(self, func_name, args, token) -> None:
        self.name = func_name
        self.args = args
        self.token = token

    def __repr__(self) -> str:
        return f"{self.name}({self.args})"

class ListNode:
    def __init__(self, expressions:list) -> None:
        self.exprs = expressions
    
    def __repr__(self) -> str:
        return f"{self.exprs}"
        
#############################
# ParseResult
#############################

class ParseResult:
    def __init__(self) -> None:
        self.error = None
        self.node = None
    
    def register(self, res):
        if isinstance(res, ParseResult):
            if res.error: self.error = res.error
            else: return res.node
        
        return res
    
    def success(self, node):
        self.node = node
        return self
    
    def failure(self, error:Error):
        self.error = error
        return self

#############################
# Parser
#############################

class Parser:
    def __init__(self, tokens:list[Token]) -> None:
        self.tokens = tokens
        self.idx = 0
        self.current_token = self.tokens[self.idx]
    
    def advance(self):
        self.idx += 1
        if self.idx < len(self.tokens):
            self.current_token = self.tokens[self.idx]
        return self.current_token

    def parse(self):
        result = self.statements()
            
        return result

    def for_expr(self):
        res = ParseResult()
        expressions = []
        self.advance()

        if not self.current_token.matches(TT_KEYWORD, "var"):
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected 'var'",
            self.current_token.pos_start.idx, self.current_token.pos_end.idx
        ))
        self.advance()

        if not self.current_token.type == TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected IDENTIFIER",
            self.current_token.pos_start.idx, self.current_token.pos_end.idx
        ))

        var_name = self.current_token
        self.advance()

        if not self.current_token.matches(TT_KEYWORD, "in"):
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected 'in'",
            self.current_token.pos_start.idx, self.current_token.pos_end.idx
        ))
        self.advance()

        expression = res.register(self.expr())
        if res.error: return res

        if not self.current_token.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected 'then'",
            self.current_token.pos_start.idx, self.current_token.pos_end.idx
        ))
        self.advance()

        while self.current_token != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
                
            expr = res.register(self.expr())
            if res.error: return res
            expressions.append(expr)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        return res.success(ForNode(var_name, expression, expressions))

    def while_expr(self):
        res = ParseResult()
        expressions = []

        self.advance()
        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_token.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected 'then'",
                self.current_token.pos_start.idx, self.current_token.pos_end.idx
            ))

        self.advance()

        while self.current_token != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
                
            expr = res.register(self.expr())
            if res.error: return res
            expressions.append(expr)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        return res.success(WhileNode(condition, expressions))

    def if_expr(self):
        res = ParseResult()
        expressions = []
        else_ = []
        else_case = False

        self.advance()
        condition = res.register(self.expr())
        if res.error: return res

        if not self.current_token.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected 'then'",
                self.current_token.pos_start.idx, self.current_token.pos_end.idx
            ))
        
        self.advance()

        while self.current_token.type != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            if self.current_token.matches(TT_KEYWORD, "fi"): 
                self.advance()
                break
            if self.current_token.matches(TT_KEYWORD, "else"):
                self.advance()
                else_case = True
                break

            expr = res.register(self.expr())
            if res.error: return res
            expressions.append(expr)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        if else_case:
            while self.current_token.type != TT_EOF:
                if self.current_token.type == TT_NEWLINE:
                    self.advance()
                    continue
                if self.current_token.matches(TT_KEYWORD, "fi"): 
                    self.advance()
                    break

                expr = res.register(self.expr())
                if res.error: return res
                else_.append(expr)

                if self.current_token.type != TT_NEWLINE: break
                self.advance()

        return res.success(IfNode(condition, expressions, else_))

    def func_expr(self):
        res = ParseResult()
        arguments = []
        expressions = []

        self.advance()
        if self.current_token.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected Identifier",
                self.current_token.pos_start.idx, self.current_token.pos_end.idx
            ))
        func_name = self.current_token
        token = Token(type=TT_FUNC, value=self.current_token.value, pos_start=self.current_token.pos_start)
        self.advance()

        if self.current_token.type != TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected '('",
                self.current_token.pos_start.idx, self.current_token.pos_end.idx
            ))
        self.advance()

        if self.current_token.type == TT_IDENTIFIER:
            arguments.append(self.current_token)
            self.advance()

        while self.current_token.type == TT_COMMA:
            self.advance()
            
            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected Identifier",
                    self.current_token.pos_start.idx, self.current_token.pos_end.idx
                ))

            arguments.append(self.current_token)
            self.advance()

        if self.current_token.type != TT_RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ',' or ')'",
                self.current_token.pos_start.idx, self.current_token.pos_end.idx
            ))
        token.pos_end = self.current_token.pos_end
        self.advance()

        if not self.current_token.matches(TT_KEYWORD, 'then'):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected 'then'",
                self.current_token.pos_start.idx, self.current_token.pos_end.idx
            ))
        self.advance()

        while self.current_token != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
                
            expr = res.register(self.expr())
            if res.error: return res
            expressions.append(expr)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        return res.success(FuncDefNode(func_name, arguments, expressions, token))

    def atom(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_INT, TT_FLOAT):
            self.advance()

            if self.current_token.type == TT_LSQUARE:
                self.advance()
                index = res.register(self.expr())

                if self.current_token.type != TT_RSQUARE:
                    return res.failure(TypeError(
                        self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                        "Expected ']'",
                        self.current_token.pos_start.idx, self.current_token.pos_end.idx
                    ))
                self.advance()
                return res.success(IndexAccessNode(NumNode(token), index))
            return res.success(NumNode(token))
        
        if token.type == TT_STR:
            self.advance()

            if self.current_token.type == TT_LSQUARE:
                self.advance()
                index = res.register(self.expr())

                if self.current_token.type != TT_RSQUARE:
                    return res.failure(TypeError(
                        self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                        "Expected ']'",
                        self.current_token.pos_start.idx, self.current_token.pos_end.idx
                    ))
                self.advance()
                return res.success(IndexAccessNode(StrNode(token), index))

            return res.success(StrNode(token))

        if token.type == TT_IDENTIFIER:
            self.advance()

            if self.current_token.type == TT_LSQUARE:
                self.advance()
                index = res.register(self.expr())

                if self.current_token.type != TT_RSQUARE:
                    return res.failure(TypeError(
                        self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                        "Expected ']'",
                        self.current_token.pos_start.idx, self.current_token.pos_end.idx
                    ))
                self.advance()
                return res.success(IndexAccessNode(VarNode(token), index))

            if self.current_token.type == TT_LPAREN:
                token_ = Token(type=TT_FUNC, value=token.value, pos_start=token.pos_start)
                args = []

                self.advance()
                if self.current_token.type != TT_RPAREN:
                    arg = res.register(self.expr())
                    if res.error: return res

                    args.append(arg)
                    while self.current_token.type == TT_COMMA:
                        self.advance()
                        arg = res.register(self.expr())
                        if res.error: return res

                        args.append(arg)
                    
                if self.current_token.type != TT_RPAREN:
                    return res.failure(TypeError(
                        self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                        "Expected ')'",
                        self.current_token.pos_start.idx, self.current_token.pos_end.idx
                    ))
                token_.pos_end = self.current_token.pos_end
                self.advance()

                return res.success(CallNode(VarNode(token), args, token_))

            return res.success(VarNode(token))
        
        if token.type == TT_LPAREN:
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_token.type == TT_RPAREN:
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
					self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
					"Expected ')'", 
                    self.current_token.pos_start.idx, self.current_token.pos_end.idx
				))

        if self.current_token.matches(TT_KEYWORD, 'if'):
            if_expr = res.register(self.if_expr())
            if res.error: return res

            return res.success(if_expr)

        if self.current_token.matches(TT_KEYWORD, 'while'):
            while_expr = res.register(self.while_expr())
            if res.error: return res

            return res.success(while_expr)

        if self.current_token.matches(TT_KEYWORD, 'for'):
            for_expr = res.register(self.for_expr())
            if res.error: return res

            return res.success(for_expr)

        if self.current_token.matches(TT_KEYWORD, 'def'):
            func_expr = res.register(self.func_expr())
            if res.error: return res

            return res.success(func_expr)

        return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected int, float, str, '+', '-', Identifier, '(', or '['",
            self.current_token.pos_start.idx, self.current_token.pos_end.idx
        ))

    def power(self):
        return self.Bin_Op(self.atom, (TT_POW, ), self.factor)

    def factor(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MIN):
            self.advance()
            factor = res.register(self.factor())
            if res.error: return res
            return res.success(UnaryOpNode(token, factor))
            
        return self.power()

    def term(self):
        return self.Bin_Op(self.factor, (TT_MUL, TT_DIV))

    def arith_expr(self):
        return self.Bin_Op(self.term, (TT_PLUS, TT_MIN))

    def comp_expr(self):
        res = ParseResult()
        if self.current_token.matches(TT_KEYWORD, 'not'):
            token = self.current_token
            self.advance()

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(token, node))
        
        node = res.register(self.Bin_Op(self.arith_expr, (TT_IS, TT_ISNOT, TT_LT, TT_GT, TT_LTE, TT_GTE)))
    
        if res.error: return res
        return res.success(node)

    def expr(self):
        res = ParseResult()
        if self.current_token.matches(TT_KEYWORD, 'var'):
            self.advance()

            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected Identifier",
                    self.current_token.pos_start.idx, self.current_token.pos_end.idx
                ))
            
            var_name = self.current_token
            self.advance()

            if self.current_token.type != TT_EQ:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected '='",
                    self.current_token.pos_start.idx, self.current_token.pos_end.idx
                ))
            
            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.Bin_Op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))
        if res.error: return res
        return res.success(node)

    def statements(self):
        res = ParseResult()
        statements = []

        while self.current_token.type == TT_NEWLINE:
            self.advance()
        
        statement = res.register(self.expr())
        if res.error: return res
        statements.append(statement)
        more_statements = True

        while self.current_token.type != TT_EOF:

            if self.current_token.type != TT_NEWLINE:
                return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Token cannot appear after previous token", 
                self.current_token.pos_start.idx, self.current_token.pos_end.idx
            ))

            newline_count = 0
            while self.current_token and self.current_token.type == TT_NEWLINE:
                self.advance()
                newline_count += 1
                if newline_count == 0:
                    more_statements = False
            
            if self.current_token.type == TT_EOF: break
            if not more_statements: break
            if not self.current_token: break

            statement = res.register(self.expr())
            statements.append(statement)
        
        return res.success(ListNode(statements))

    def Bin_Op(self, function, operators, function2=None):
        res = ParseResult()
        if not function2:
            function2 = function

        left = res.register(function())
        if res.error: return res
        
        while self.current_token.type in operators or (self.current_token.type, self.current_token.value) in operators:
            token = self.current_token
            self.advance()
            right = res.register(function2())
            if res.error: return res
            left = BinOpNode(left, token, right)
        
        return res.success(left)

#######################################
# RunTimeResult
#######################################

class RunTimeResult:
    def __init__(self) -> None:
        self.value = None
        self.error = None
        
    def register(self, res):
        if isinstance(res, RunTimeResult):
            if res.error: self.error = res.error
            else: return res.value
        return res

    def success(self, value):
        self.value = value
        return self
    
    def failure(self, error):
        self.error = error
        return self

#######################################
# Values
#######################################

class Value:
    def __init__(self, node:Value|Token) -> None:
        if type(node) == Token:
            self.token = node
        else:
            self.token = node.token
    
    def IllegalOperation(self, other=None, operand:str=None):
        res = RunTimeResult()
        if not other: other = self

        return res.failure(TypeError(
            self.token.pos_start.fn, self.token.pos_start.current_line,
            f"Unsupported operand type(s) for {operand}: {self.token.type.lower()} and {other.token.type.lower()}",
            self.token.pos_start.idx, other.token.pos_end.idx
        ))

    def plus(self, other:Value):
        return self.IllegalOperation(other, '+')
    
    def minus(self, other:Value):
        return self.IllegalOperation(other, '-')

    def multiply(self, other:Value):
        return self.IllegalOperation(other, '*')

    def divide(self, other:Value):
        return self.IllegalOperation(other, '/')
    
    def power(self, other:Value):
        return self.IllegalOperation(other, '^')

    def equals(self, other:Value):
        return self.IllegalOperation(other, '==')

    def not_equals(self, other:Number):
        return self.IllegalOperation(other, '!=')

    def greater_than(self, other:Number):
        return self.IllegalOperation(other, '>')

    def greater_equals(self, other:Number):
        return self.IllegalOperation(other, '>=')
    
    def lower_than(self, other:Number):
        return self.IllegalOperation(other, '<')

    def lower_equals(self, other:Number):
        return self.IllegalOperation(other, '<=')

    def and_(self, other:Number):
        return self.IllegalOperation(other, 'and')

    def or_(self, other:Number):
        return self.IllegalOperation(other, 'or')

    def not_(self):
        return self.IllegalOperation(self, 'not')

    def is_true(self):
        return False

    def index(self, other:Value):
        return self.IllegalOperation(other, '')

    def __repr__(self) -> str:
        return f"{self.token}"

class Number(Value):
    def __init__(self, node:NumNode|Token) -> None:
        super().__init__(node)
        self.token.value = int(self.token.value) if self.token.type == TT_INT else float(self.token.value)
    
    def plus(self, other:Number):
        res = RunTimeResult()

        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '+')

        type=self.token.type
        value = eval(f"{self.token.value}+{other.token.value}")       
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def minus(self, other:Number):
        res = RunTimeResult()
        
        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '-')

        type=self.token.type        
        value = eval(f"{self.token.value}-{other.token.value}")               
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))

    def multiply(self, other:Number):
        res = RunTimeResult()

        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '*')

        type=self.token.type
        value = eval(f"{self.token.value}*{other.token.value}")
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def divide(self, other:Number):
        res = RunTimeResult()

        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '/')

        if other.token.value == 0:
            return res.failure(ZeroDivisionError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                "Can't divide by zero", 
                self.token.pos_start.idx, other.token.pos_end.idx
            ))

        type=self.token.type
        value = eval(f"{self.token.value}/{other.token.value}")

        type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def power(self, other:Number):
        res = RunTimeResult()
        
        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '^')

        type=self.token.type
        value = eval(f"{self.token.value}**{other.token.value}")

        if other.token.type == TT_FLOAT or other.token.value < 0:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))

    def equals(self, other:Number):
        res = RunTimeResult()
        if self.token.matches(token=other.token): value = "1"
        else: value = "0"
        return res.success(Number(Token(type=TT_INT, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def not_equals(self, other:Number):
        res = RunTimeResult()
        if self.token.matches(token=other.token): value = "0"
        else: value = "1"
        return res.success(Number(Token(type=TT_INT, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def greater_than(self, other:Number):
        res = RunTimeResult()

        if not isinstance(other, (NumNode, Number)):
            return self.IllegalOperation(other, '>')
        return res.success(Number(Token(type=TT_INT, value="1" if float(self.token.value) > float(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def greater_equals(self, other:Number):
        res = RunTimeResult()

        if not isinstance(other, (NumNode, Number)):
            return self.IllegalOperation(other, '>=')
        return res.success(Number(Token(type=TT_INT, value="1" if float(self.token.value) >= float(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))
    
    def lower_than(self, other:Number):
        res = RunTimeResult()

        if not isinstance(other, (NumNode, Number)):
            return self.IllegalOperation(other, '<')
        return res.success(Number(Token(type=TT_INT, value="1" if float(self.token.value) < float(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def lower_equals(self, other:Number):
        res = RunTimeResult()

        if not isinstance(other, (NumNode, Number)):
            return self.IllegalOperation(other, '<=')
        return res.success(Number(Token(type=TT_INT, value="1" if float(self.token.value) <= float(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def and_(self, other:Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if self.is_true() and other.is_true() else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def or_(self, other:Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if self.is_true() or other.is_true() > 0 else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def not_(self):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if not self.is_true() else "0", pos_start=self.token.pos_start, pos_end=self.token.pos_end)))

    def is_true(self):
        return int(self.token.value) > 0 

    def index(self, other:Number):
        res = RunTimeResult()
        return res.failure(TypeError(
            self.token.pos_start.fn, self.token.pos_start.current_line,
            f"Type {self.token.type.lower()} is not subscriptable",
            self.token.pos_start.idx, other.token.pos_end.idx
        ))

class String(Value):
    def __init__(self, node: StrNode|Token) -> None:
        super().__init__(node)

    def __iter__(self):
        i = 0
        while i < len(self.token.value):
            yield self.token.value[i]
            i+=1

    def plus(self, other:String|StrNode):
        res = RunTimeResult()

        if other.token.type != TT_STR:
            return self.IllegalOperation(other, '+')
        
        str = self.token.value + other.token.value
        return res.success(String(Token(type=TT_STR, value=str, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))
    
    def multiply(self, other:Number|NumNode):
        res = RunTimeResult()

        if other.token.type != TT_INT:
            return self.IllegalOperation(other, '*')

        str = self.token.value * int(other.token.value)
        return res.success(String(Token(type=TT_STR, value=str, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def equals(self, other:NumNode|Number):
        res = RunTimeResult()
        if self.token.matches(token=other.token): value = "1"
        else: value = "0"
        return res.success(Number(Token(type=TT_INT, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def not_equals(self, other:NumNode|Number):
        res = RunTimeResult()
        if self.token.matches(token=other.token): value = "0"
        else: value = "1"
        return res.success(Number(Token(type=TT_INT, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def and_(self, other:Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if self.is_true() and other.is_true() else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def or_(self, other:Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if self.is_true() or other.is_true() > 0 else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def not_(self):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if not self.is_true() else "0", pos_start=self.token.pos_start, pos_end=self.token.pos_end)))

    def is_true(self):
        return len(self.token.value) > 0

    def index(self, other:Number|NumNode):
        res = RunTimeResult()
        
        if other.token.type != TT_INT:
            return res.failure(TypeError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                f"string indices must be int not '{other.token.type.lower()}'",
                self.token.pos_start.idx, other.token.pos_end.idx
            ))
        if int(other.token.value) >= len(self.token.value) or int(other.token.value) <= -len(self.token.value):
            return res.failure(IndexError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                "Str index out of range",
                self.token.pos_start.idx, other.token.pos_end.idx
            ))
        
        value = self.token.value[int(other.token.value)]
        return res.success(String(Token(type=TT_STR, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

#######################################
# Function
#######################################

class Function:
    def __init__(self, token:Token, func_args:list[Token], exprs:list) -> None:
        self.args = func_args
        self.exprs = exprs
        self.token = token
        
    def new_symbol_table(self, parent):
        symbol_table = SymbolTable(parent=parent)
        self.symbol_table = symbol_table

    def check_args(self, args):
        res = RunTimeResult()
        if len(self.args) != len(args):
            return res.failure(TypeError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                f"{self.token.value} takes {len(self.args)} arguments but {len(args)} were given",
                self.token.pos_start.idx, self.token.pos_end.idx
            ))

    def set_args(self, args):
        res = RunTimeResult()
        res.register(self.check_args(args))
        if res.error: return res

        for i, var in enumerate(self.args):
            self.symbol_table.assign(var.value, args[i])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        self.new_symbol_table(parent_symbol_table)
        res.register(self.set_args(args))
        if res.error: return res

        interpreter = Interpreter()
        for expr in self.exprs:
            res.register(interpreter.visit(expr, self.symbol_table))
            if res.error: return res

        return res.success(GlobalSymbolTable.get_variable("Null"))

    def __repr__(self) -> str:
        return f"{self.token.value}({self.args})"

#######################################
# Interpreter
#######################################

class Interpreter:
    def no_visit_method(self, node, _):
        raise AttributeError(f"no visit_{type(node).__name__} method found")

    def visit(self, node, symbol_table):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.no_visit_method)

        return method(node, symbol_table)

    def visit_ListNode(self, node:ListNode, symbol_table):
        res = RunTimeResult()
        results = []

        for statement in node.exprs:
            result = res.register(self.visit(statement, symbol_table))
            if res.error: return res
            results.append(result)
        
        return res.success(results)

    def visit_CallNode(self, node:CallNode, symbol_table):
        res = RunTimeResult()

        func:Function = res.register(self.visit(node.name, symbol_table))
        if res.error: return res

        if func.token.type != TT_FUNC:
            return res.failure(TypeError(
                func.token.pos_start.fn, func.token.pos_start.current_line,
                f"{func.token.type.lower()} is not callable",
                func.token.pos_start.idx, func.token.pos_end.idx
            ))

        args = []

        for expr in node.args:
            arg = res.register(self.visit(expr, symbol_table))
            if res.error: return res
            args.append(arg)
        
        res.register(func.execute(args, symbol_table))
        if res.error: return res

        return res.success(GlobalSymbolTable.get_variable("Null"))

    def visit_FuncDefNode(self, node:FuncDefNode, symbol_table):
        res = RunTimeResult()

        func = Function(node.token, node.args, node.exprs)
        symbol_table.assign(node.func_name.value, func)
        
        return res.success(GlobalSymbolTable.get_variable("Null"))

    def visit_ForNode(self, node:ForNode, symbol_table):
        res = RunTimeResult()
        condition = res.register(self.visit(VarAssignNode(node.var_name, node.expr), symbol_table))
        if res.error: return res

        value:Value = res.register(self.visit(node.expr, symbol_table))
        if res.error: return res

        if value.token.type not in (TT_STR,):
            return res.failure(TypeError(
                node.expr.token.pos_start.fn, node.expr.token.pos_start.current_line,
                f"{node.expr.token.type.lower()} is not iterable",
                node.expr.token.pos_start.idx, node.expr.token.pos_end.idx
            ))

        for i in value:
            condition = res.register(self.visit(VarAssignNode(node.var_name, StrNode(Token(type=TT_STR, value = i))), symbol_table))
            if res.error: return res
            for expr in node.exprs:
                res.register(self.visit(expr, symbol_table))
                if res.error: return res
        
        return res.success(condition)

    def visit_WhileNode(self, node:WhileNode, symbol_table):
        res = RunTimeResult()

        condition:Number = res.register(self.visit(node.condition, symbol_table))
        if res.error: return res

        while condition.is_true():
            for expr in node.exprs:
                res.register(self.visit(expr, symbol_table))
                if res.error: return res
            condition = res.register(self.visit(node.condition, symbol_table))
        
        return res.success(condition)

    def visit_IfNode(self, node:IfNode, symbol_table):
        res = RunTimeResult()

        condition:Value = res.register(self.visit(node.condition, symbol_table))
        if res.error: return res

        if condition.is_true():
            for expr in node.exprs:
                expr = res.register(self.visit(expr, symbol_table))
                if res.error: return res

        else:
            for expr in node.else_:
                expr = res.register(self.visit(expr, symbol_table))
                if res.error: return res
        
        return res.success(condition)
    
    def visit_IndexAccessNode(self, node:IndexAccessNode, symbol_table):
        res = RunTimeResult()
        
        value:Value = res.register(self.visit(node.token, symbol_table))
        if res.error: return res
        index:Number = res.register(self.visit(node.index, symbol_table)) 
        if res.error: return res

        char = res.register(value.index(index))
        return res.success(char)

    def visit_VarNode(self, node:VarNode, symbol_table):
        res = RunTimeResult()
        var_name = node.var.value
        value = symbol_table.get_variable(var_name)

        if not value:
            return res.failure(NameError(
                node.var.pos_start.fn, node.var.pos_start.current_line,
                f"{var_name} is not defined",
                node.var.pos_start.idx, node.var.pos_end.idx
            ))
        
        value.token.pos_start = node.var.pos_start
        value.token.pos_end = node.var.pos_end
        return res.success(value)
    
    def visit_VarAssignNode(self, node:VarAssignNode, symbol_table):
        res = RunTimeResult()
        var_name = node.var_name.value
        value = res.register(self.visit(node.value, symbol_table))
        if res.error: return res

        symbol_table.assign(var_name, value)
        return res.success(value)

    def visit_StrNode(self, node:StrNode, symbol_table):
        res = RunTimeResult()
        node = String(node)
        return res.success(node)

    def visit_NumNode(self, node:NumNode, symbol_table):
        res = RunTimeResult()
        node = Number(node)

        return res.success(node)
    
    def visit_UnaryOpNode(self, node:UnaryOpNode, symbol_table):
        res = RunTimeResult()

        n:Value = res.register(self.visit(node.node, symbol_table))
        if res.error: return res

        pos_start = n.token.pos_start
        pos_start.back()

        if node.op_token.type == TT_MIN:
            return n.multiply(Number(Token(type=TT_INT, value="-1", pos_start=pos_start)))
        elif node.op_token.type == TT_PLUS:
            return n.multiply(Number(Token(type=TT_INT, value="1", pos_start=pos_start)))
        elif node.op_token.matches(TT_KEYWORD, 'not'):
            return n.not_()

    def visit_BinOpNode(self, node:BinOpNode, symbol_table):
        res = RunTimeResult()

        left:Value = res.register(self.visit(node.left, symbol_table))
        if res.error: return res
        right:Value = res.register(self.visit(node.right, symbol_table))
        if res.error: return res
        
        ops = {
            TT_PLUS: left.plus,
            TT_MIN: left.minus,
            TT_MUL: left.multiply,
            TT_DIV: left.divide,
            TT_POW: left.power,
            TT_IS: left.equals,
            TT_ISNOT: left.not_equals,
            TT_GT: left.greater_than,
            TT_GTE: left.greater_equals,
            TT_LT: left.lower_than,
            TT_LTE: left.lower_equals,
        }

        if node.op.type == TT_KEYWORD: 
            if node.op.value == 'and':
                result = res.register(left.and_(right))
            if node.op.value == 'or':
                result = res.register(left.or_(right))
        elif node.op.type in ops:
            result = res.register(ops[node.op.type](right))
        
        if res.error: return res
        return res.success(result)

GlobalSymbolTable = SymbolTable()
GlobalSymbolTable.add_keywords(
    'var', 'def',
    'and', 'or', 'not', 
    'if', 'then', 'else', 'fi', 
    'while', 'for', 'in', 'end'
)
GlobalSymbolTable.assign_multiple(
    ("Null", Number(Token(type=TT_INT, value=0))), 
    ("True", Number(Token(type=TT_INT, value=1))),
    ("False", Number(Token(type=TT_INT, value=0))),
    ("PI", Number(Token(type=TT_FLOAT, value=str(math.pi))))
)

def Main(input, fn):
    lexer = Lexer(input, fn)
    tokens, error = lexer.make_tokens()
    if error: return tokens, error
    if len(tokens) == 1: return [], None

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    result = interpreter.visit(ast.node, GlobalSymbolTable) 

    return result.value, result.error