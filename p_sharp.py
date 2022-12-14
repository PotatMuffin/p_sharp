from __future__ import annotations
from arrows import add_arrows
from os import path
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
    def __init__(self, fn, line:str, error_name:str, details:str, pos_start:Position, pos_end:Position=None) -> None:
        self.fn = fn
        self.line = line
        self.error = error_name
        self.details = details
        self.pos_start = pos_start
        self.pos_end = pos_end

        if not pos_end:
            self.pos_end = pos_start

    def as_string(self):
        return f"{self.error}: '{self.details}'\nFile: {self.fn}, Line: {self.pos_start.ln}\n\n{add_arrows(self.line, self.pos_start.idx, self.pos_end.idx)}"

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

class KeyError(Error):
    def __init__(self, fn, line: str, details: str, pos_start: Position, pos_end: Position = None) -> None:
        super().__init__(fn, line, "Key Error", details, pos_start, pos_end)

class AttributeError_(Error):
    def __init__(self, fn, line: str, details: str, pos_start: Position, pos_end: Position = None) -> None:
        super().__init__(fn, line, "Attribute error", details, pos_start, pos_end)

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
            value = self.parent.get_variable(variable)
        return value
    
    def add_keywords(self, *kw):
        for keyword in kw:
            self.keywords.append(keyword)

    def assign(self, var_name, value):
        self.symbols[var_name] = value
    
    def assign_multiple(self, *vars:tuple[tuple]):
        for var in vars:
            self.symbols[var[0]] = var[1]

    def copy(self):
        new_table = SymbolTable()
        new_table.symbols = self.symbols.copy()
        new_table.parent = self.parent
        return new_table

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
TT_POW = "POW"

TT_COMMA = "COMMA"
TT_COLON = "COLON"
TT_DOT = "DOT"

TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"
TT_LSQUARE = "LSQUARE"
TT_RSQUARE = "RSQUARE"
TT_LBRACKET = "LBRACKET"
TT_RBRACKET = "RBRACKET"

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_STR = "STR"
TT_LIST = "LIST"
TT_DICT = "DICT"
TT_FUNC = "FUNC"
TT_METHOD = "METHOD"
TT_CLASS = "CLASS"

TT_IDENTIFIER = "IDENTIFIER"
TT_KEYWORD = "KEYWORD"

TT_EQ = "EQ"
TT_PLUSEQ = "PLUSEQ"
TT_MINEQ = "MINEQ"
TT_MULEQ = "MULEQ"
TT_DIVEQ = "DIVEQ"
TT_POWEQ = "POWEQ"
TT_IS = "IS"
TT_ISNOT = "ISNOT"
TT_GT = "GT" # greater than
TT_GTE = "GTE" # greater than or equals
TT_LT = "LT" # lower than
TT_LTE = "LTE" # lower than or equals

TT_NEWLINE = "NEWLINE"
TT_EOF = "EOF"

class Token:
    def __init__(self, type, value=None, pos_start:Position=Position(0,0,"",""), pos_end:Position=None) -> None:
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
    
    def make_equals(self, token):
        tokens = {
            '-': TT_MIN, '-=': TT_MINEQ,
            '+': TT_PLUS, '+=': TT_PLUSEQ,
            '*': TT_MUL, '*=': TT_MULEQ,
            '/': TT_DIV, '/=': TT_DIVEQ,
            '^': TT_POW, '^=': TT_POWEQ,
            '=': TT_EQ, '==': TT_IS
        }

        pos = self.pos.copy()
        self.pos.advance()
        if self.pos.current_char == '=':
            token += '='
            return Token(type=tokens[token], pos_start=pos, pos_end=self.pos.copy())
        self.pos.back()
        return Token(type=tokens[token], pos_start=pos)
    
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
        quote = self.pos.current_char
        escape_character = False

        escaped_character = {
            "n": "\n",
            "t": "\t"
        }

        self.pos.advance()
        while self.pos.current_char and self.pos.current_char in characters:
            if escape_character:
                string += escaped_character.get(self.pos.current_char, self.pos.current_char)
                escape_character = False
            else:
                if self.pos.current_char == quote: 
                    self.pos.advance()
                    break

                if self.pos.current_char == "\\":
                    escape_character = True
                else:
                    string += self.pos.current_char
            self.pos.advance()
        
        pos_end = self.pos.copy()
        pos_end.back()

        return Token(type=TT_STR, value=string, pos_start=pos, pos_end=pos_end)

    def make_tokens(self):
        tokens = []
        
        chars = {
        '(': lambda: tokens.append(Token(type=TT_LPAREN, pos_start=self.pos.copy())),
        ')': lambda: tokens.append(Token(type=TT_RPAREN, pos_start=self.pos.copy())),
        '[': lambda: tokens.append(Token(type=TT_LSQUARE, pos_start=self.pos.copy())),
        ']': lambda: tokens.append(Token(type=TT_RSQUARE, pos_start=self.pos.copy())),
        '{': lambda: tokens.append(Token(type=TT_LBRACKET, pos_start=self.pos.copy())),
        '}': lambda: tokens.append(Token(type=TT_RBRACKET, pos_start=self.pos.copy())),
        ';': lambda: tokens.append(Token(type=TT_NEWLINE, pos_start=self.pos.copy())),
        ',': lambda: tokens.append(Token(type=TT_COMMA, pos_start=self.pos.copy())),
        ':': lambda: tokens.append(Token(type=TT_COLON, pos_start=self.pos.copy())),
        '.': lambda: tokens.append(Token(type=TT_DOT, pos_start=self.pos.copy())),
        '+': lambda: tokens.append(self.make_equals('+')), 
        '-': lambda: tokens.append(self.make_equals('-')), 
        '*': lambda: tokens.append(self.make_equals('*')), 
        '/': lambda: tokens.append(self.make_equals('/')),
        '^': lambda: tokens.append(self.make_equals('^')),
        '=': lambda: tokens.append(self.make_equals('=')),
        '!': lambda: tokens.append(self.make_not_equals()),
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

                if self.pos.current_char == 'e':
                    chars["*"]()
                    tokens.append(Token(type=TT_INT, value=10, pos_start=self.pos.copy()))
                    chars["^"]()
                    self.pos.advance()
                    
                    if self.pos.current_char and self.pos.current_char in DIGITS:
                        tokens.append(self.make_digit())
                    else: return [], InvalidSyntaxError(self.pos.fn, self.pos.current_line, self.pos.current_char, self.pos)
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
                return [], IllegalCharError(self.pos.fn, self.pos.current_line, self.pos.current_char, self.pos)

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
    def __init__(self, var_name:Token, value, Global=False) -> None:
        self.var_name = var_name
        self.value = value
        self.Global = Global

    def __repr__(self) -> str:
        return f"({self.var_name} = {self.value})"

class VarNode:
    def __init__(self, var:Token) -> None:
        self.var = var
    
    def __repr__(self) -> str:
        return f"{self.var}"

class AttributeAssignNode:
    def __init__(self, class_name, attribute, value) -> None:
        self.class_name = class_name
        self.attribute = attribute
        self.value = value
    
    def __repr__(self) -> str:
        return f"{self.class_name}.{self.attribute} = {self.value}"

class AttributeNode:
    def __init__(self, class_name, attribute) -> None:
        self.class_name = class_name
        self.attribute = attribute
    
    def __repr__(self) -> str:
        return f"{self.class_name}.{self.attribute}"

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

class TryNode:
    def __init__(self, exprs, except_exprs, finally_exprs) -> None:
        self.exprs = exprs
        self.except_exprs = except_exprs
        self.finally_exprs = finally_exprs

    def __repr__(self) -> str:
        return f"try: {self.exprs} \nexcept: {self.except_exprs} \nfinally: {self.finally_exprs}"

class FuncDefNode:
    def __init__(self, func_name:Token, arguments:list[tuple[Token, Token|None]], expressions:list, token) -> None:
        self.func_name = func_name
        self.args = arguments
        self.exprs = expressions
        self.token = token
    
    def __repr__(self) -> str:
        return f"def {self.func_name}({self.args}): {self.exprs}"

class ClassNode:
    def __init__(self, class_name:Token, statements:list, parent:Token=None) -> None:
        self.class_name = class_name
        self.statements = statements
        self.parent = parent
    
    def __repr__(self) -> str:
        return f"class {self.class_name}{f'({self.inheritance}):' if self.inheritance else ':'}"

class CallNode:
    def __init__(self, func_name, args, token, cls=None) -> None:
        self.name = func_name
        self.args = args
        self.token = token
        self.cls = cls

    def __repr__(self) -> str:
        return f"{self.name}({self.args})"

class ListNode:
    def __init__(self, expressions:list) -> None:
        self.exprs = expressions
    
    def __repr__(self) -> str:
        return f"{self.exprs}"

class DictNode:
    def __init__(self, elements:list[tuple]) -> None:
        self.elements = elements
    
    def __repr__(self) -> str:
        return f"{self.elements}"

class ReturnNode:
    def __init__(self, token:Token, value) -> None:
        self.token = token
        self.return_value = value
    
    def __repr__(self) -> str:
        return f"return {self.return_value}"

class BreakNode:
    def __init__(self, token:Token) -> None:
        self.token = token
    
    def __repr__(self) -> str:
        f"{self.token}"

class ContinueNode:
    def __init__(self, token:Token) -> None:
        self.token = token

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

    def try_expr(self):
        res = ParseResult()
        expressions = []
        except_expressions = []
        except_case = False
        finally_expressions = []
        finally_case = False
        self.advance()

        if self.current_token.type != TT_COLON:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ':'",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        self.advance()

        while self.current_token.type != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue       
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
            if self.current_token.matches(TT_KEYWORD, "except"):
                except_case = True
                self.advance()
                break
            if self.current_token.matches(TT_KEYWORD, "finally"):
                finally_case = True
                self.advance()
                break

            expr = res.register(self.statement())
            if res.error: return res

            expressions.append(expr)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        if except_case:
            if self.current_token.type != TT_COLON:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected ':'",
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            self.advance()
            while self.current_token.type != TT_EOF:
                if self.current_token.type == TT_NEWLINE:
                    self.advance()
                    continue
                if self.current_token.matches(TT_KEYWORD, "end"):
                    self.advance()
                    break
                if self.current_token.matches(TT_KEYWORD, "finally"):
                    finally_case = True
                    self.advance()
                    break

                expr = res.register(self.statement())
                if res.error: return res

                except_expressions.append(expr)
                if self.current_token.type != TT_NEWLINE: break
                self.advance()
            
        if finally_case:
            if self.current_token.type != TT_COLON:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected ':'",
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            self.advance()
            while self.current_token.type != TT_EOF:
                if self.current_token.type == TT_NEWLINE:
                    self.advance()
                    continue
                if self.current_token.matches(TT_KEYWORD, "end"):
                    self.advance()
                    break
                if self.current_token.matches(TT_KEYWORD, "except"):
                    return res.failure(InvalidSyntaxError(
                        self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                        "'except' can't appear after 'finally'",
                        self.current_token.pos_start, self.current_token.pos_end
                    ))

                expr = res.register(self.statement())
                if res.error: return res

                finally_expressions.append(expr)
                if self.current_token.type != TT_NEWLINE: break
                self.advance()
        
        return res.success(TryNode(expressions, except_expressions, finally_expressions))

    def class_expr(self):
        res = ParseResult()
        parent = None
        statements = []
        self.advance()

        if self.current_token.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected Identifier",
                self.current_token.pos_start, self.current_token.pos_end
            ))

        class_name = self.current_token
        class_name.type = TT_CLASS
        self.advance()

        if self.current_token.type == TT_LPAREN:
            self.advance()
            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected Identifier", 
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            
            parent = VarNode(self.current_token)
            self.advance()

            if self.current_token.type != TT_RPAREN:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected ')",
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            self.advance()
        
        if self.current_token.type != TT_COLON:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ':'",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        self.advance()

        while self.current_token.type != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
                
            statement = res.register(self.statement())
            if res.error: return res
            statements.append(statement)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        return res.success(ClassNode(class_name, statements, parent))

    def for_expr(self):
        res = ParseResult()
        expressions = []
        self.advance()

        if not self.current_token.matches(TT_KEYWORD, "var"):
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected 'var'",
            self.current_token.pos_start, self.current_token.pos_end
        ))
        self.advance()

        if not self.current_token.type == TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected IDENTIFIER",
            self.current_token.pos_start, self.current_token.pos_end
        ))

        var_name = self.current_token
        self.advance()

        if not self.current_token.matches(TT_KEYWORD, "in"):
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected 'in'",
            self.current_token.pos_start, self.current_token.pos_end
        ))
        self.advance()

        expression = res.register(self.statement())
        if res.error: return res

        if not self.current_token.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected 'then'",
            self.current_token.pos_start, self.current_token.pos_end
        ))
        self.advance()

        while self.current_token != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
                
            expr = res.register(self.statement())
            if res.error: return res
            expressions.append(expr)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        return res.success(ForNode(var_name, expression, expressions))

    def while_expr(self):
        res = ParseResult()
        expressions = []

        self.advance()
        condition = res.register(self.statement())
        if res.error: return res

        if not self.current_token.matches(TT_KEYWORD, "then"):
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected 'then'",
                self.current_token.pos_start, self.current_token.pos_end
            ))

        self.advance()

        while self.current_token != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
                
            expr = res.register(self.statement())
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
                self.current_token.pos_start, self.current_token.pos_end
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

            expr = res.register(self.statement())
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

                expr = res.register(self.statement())
                if res.error: return res
                else_.append(expr)

                if self.current_token.type != TT_NEWLINE: break
                self.advance()

        return res.success(IfNode(condition, expressions, else_))

    def func_expr(self):
        res = ParseResult()
        should_have_default_value = False
        arguments = []
        expressions = []

        self.advance()
        if self.current_token.type != TT_IDENTIFIER:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected Identifier",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        func_name = self.current_token
        token = Token(type=TT_FUNC, value=self.current_token.value, pos_start=self.current_token.pos_start)
        self.advance()

        if self.current_token.type != TT_LPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected '('",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        self.advance()

        if self.current_token.type == TT_IDENTIFIER:
            token_ = self.current_token
            default_value = None
            self.advance()
            if self.current_token.type == TT_EQ:
                should_have_default_value = True
                self.advance()
                default_value = res.register(self.expr())
                if res.error: return res
            
            arguments.append((token_, default_value))

        while self.current_token.type == TT_COMMA:
            self.advance()
            
            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected Identifier",
                    self.current_token.pos_start, self.current_token.pos_end
                ))

            token_ = self.current_token
            default_value = None
            self.advance()
            if self.current_token.type == TT_EQ:
                should_have_default_value = True
                self.advance()
                default_value = res.register(self.expr())
                if res.error: return res
            elif should_have_default_value:
                return res.failure(InvalidSyntaxError(
                    token_.pos_start.fn, token_.pos_start.current_line,
                    f"Non-default argument can not follow a default argument",
                    token_.pos_start, token_.pos_end
                ))

            arguments.append((token_, default_value))

        if self.current_token.type != TT_RPAREN:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ',' or ')'",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        token.pos_end = self.current_token.pos_end
        self.advance()

        if self.current_token.type != TT_COLON:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ':'",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        self.advance()

        while self.current_token != TT_EOF:
            if self.current_token.type == TT_NEWLINE:
                self.advance()
                continue
            if self.current_token.matches(TT_KEYWORD, "end"):
                self.advance()
                break
                
            expr = res.register(self.statement())
            if res.error: return res
            expressions.append(expr)

            if self.current_token.type != TT_NEWLINE: break
            self.advance()
        
        return res.success(FuncDefNode(func_name, arguments, expressions, token))

    def dict(self):
        res = ParseResult()
        self.advance()
        
        while self.current_token.type == TT_NEWLINE:
            self.advance()

        elements = []
        if self.current_token.type != TT_RBRACKET:
            key = res.register(self.expr())
            if res.error: return res
        
            if self.current_token.type != TT_COLON:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected ':'",
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            self.advance()

            value = res.register(self.expr())
            if res.error: return res
            elements.append((key, value))
        
        while self.current_token.type == TT_COMMA:
            self.advance()

            while self.current_token.type == TT_NEWLINE:
                self.advance()
            
            key = res.register(self.expr())
            if res.error: return res

            if self.current_token.type != TT_COLON:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected ':'",
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            self.advance()

            value = res.register(self.expr())
            if res.error: return res
            elements.append((key, value))
        self.advance()

        return res.success(DictNode(elements))

    def list(self):
        res = ParseResult()
        self.advance()
        
        while self.current_token.type == TT_NEWLINE:
            self.advance()

        expressions = []
        if self.current_token.type != TT_RSQUARE:
            expr = res.register(self.expr())
            if res.error: return res
            expressions.append(expr)

        while self.current_token.type == TT_COMMA:
            self.advance()

            while self.current_token.type == TT_NEWLINE:
                self.advance()

            expr = res.register(self.expr())
            if res.error: return res
            expressions.append(expr)

        while self.current_token.type == TT_NEWLINE:
            self.advance()

        if self.current_token.type != TT_RSQUARE:
            return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ',' or ']'",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        self.advance()

        return res.success(ListNode(expressions))

    def index(self, node):
        res = ParseResult()
        self.advance()
        index = res.register(self.statement())

        if self.current_token.type != TT_RSQUARE:
            return res.failure(TypeError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ',' or ']'",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        self.advance()
        return res.success(IndexAccessNode(node, index))

    def access_attribute(self, token, node=None):
        res = ParseResult()
        n = node

        self.advance()
        attribute = self.current_token
        node = AttributeNode(node or VarNode(token), attribute)
        if not n: n = node

        self.advance()
        if self.current_token.type == TT_LPAREN:
            node = res.register(self.call(token, node))
            if res.error: return res
            node.cls = n if type(n) != AttributeNode else n.class_name

        if self.current_token.type == TT_DOT:
            node = res.register(self.access_attribute(token, node))
            if res.error: return res

        if self.current_token.type == TT_LSQUARE:
            node = res.register(self.index(node))
        
        if self.current_token.type in (TT_EQ, TT_DIVEQ, TT_MINEQ, TT_MULEQ, TT_PLUSEQ, TT_POWEQ):
            op = self.current_token
            self.advance()
            value = res.register(self.expr())
            if res.error: return res

            if op.type in (TT_DIVEQ, TT_MINEQ, TT_MULEQ, TT_PLUSEQ, TT_POWEQ):
                BinOps = {TT_DIVEQ: TT_DIV, TT_MINEQ: TT_MIN, TT_MULEQ: TT_MUL, TT_PLUSEQ: TT_PLUS, TT_POWEQ: TT_POW}
                BinOp = Token(type=BinOps[op.type], pos_start=self.current_token.pos_start, pos_end=self.current_token.pos_end)

                expr = BinOpNode(AttributeNode(node.class_name if type(node.class_name) == AttributeNode else VarNode(token), attribute), BinOp, value)
                node = AttributeAssignNode(node.class_name if type(node.class_name) == AttributeNode else VarNode(token), attribute, expr)
            else:
                node = AttributeAssignNode(node.class_name if type(node.class_name) == AttributeNode else VarNode(token), attribute, value)
        return res.success(node)

    def call(self, token, node=None):
        res = ParseResult()
        args = []

        self.advance()
        if self.current_token.type != TT_RPAREN:
            arg = res.register(self.statement())
            if res.error: return res

            args.append(arg)
            while self.current_token.type == TT_COMMA:
                self.advance()
                arg = res.register(self.statement())
                if res.error: return res

                args.append(arg)
            
        if self.current_token.type != TT_RPAREN:
            return res.failure(TypeError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Expected ')'",
                self.current_token.pos_start, self.current_token.pos_end
            ))
        token.pos_end = self.current_token.pos_end
        self.advance()
        
        node = CallNode(node or VarNode(token), args, token)
        
        if self.current_token.type == TT_LPAREN:
            node = res.register(self.call(node.token, node))
            if res.error: return res

        if self.current_token.type == TT_DOT:
            node = res.register(self.access_attribute(node.token, node))
            if res.error: return res

        return res.success(node)

    def atom(self):
        res = ParseResult()
        token = self.current_token

        if self.current_token.type == TT_LSQUARE:
            list = res.register(self.list())
            if res.error: return res

            if self.current_token.type == TT_LSQUARE:
                return self.index(ListNode(list.exprs))
            return res.success(list)

        if self.current_token.type == TT_LBRACKET:
            dict = res.register(self.dict())
            if res.error: return res

            if self.current_token.type == TT_LSQUARE:
                return self.index(DictNode(dict.elements))
            return res.success(dict)

        if token.type in (TT_INT, TT_FLOAT):
            self.advance()

            if self.current_token.type == TT_LSQUARE:
                return self.index(NumNode(token))
            return res.success(NumNode(token))
        
        if token.type == TT_STR:
            self.advance()

            if self.current_token.type == TT_LSQUARE:
                return self.index(StrNode(token))
            return res.success(StrNode(token))

        if token.type == TT_IDENTIFIER:
            self.advance()

            if self.current_token.type == TT_LSQUARE:
                return self.index(VarNode(token))

            if self.current_token.type == TT_DOT:
                return self.access_attribute(token)

            if self.current_token.type == TT_LPAREN:
                return self.call(token)
            return res.success(VarNode(token))

        if token.type == TT_LPAREN:
            self.advance()
            expr = res.register(self.statement())
            if res.error: return res
            if self.current_token.type == TT_RPAREN:
                self.advance()
                return res.success(expr)
            else:
                return res.failure(InvalidSyntaxError(
					self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
					"Expected ')'", 
                    self.current_token.pos_start, self.current_token.pos_end
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

        if self.current_token.matches(TT_KEYWORD, 'try'):
            try_expr = res.register(self.try_expr())
            if res.error: return res

            return res.success(try_expr)

        if self.current_token.matches(TT_KEYWORD, 'def'):
            func_expr = res.register(self.func_expr())
            if res.error: return res

            return res.success(func_expr)

        if self.current_token.matches(TT_KEYWORD, 'class'):
            class_expr = res.register(self.class_expr())
            if res.error: return res

            return res.success(class_expr)

        return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected int, float, str, '+', '-', Identifier, '(', or '['",
            self.current_token.pos_start, self.current_token.pos_end
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
        if self.current_token.matches(TT_KEYWORD, 'var') or self.current_token.matches(TT_KEYWORD, 'global'):
            Global=False

            if self.current_token.matches(TT_KEYWORD, 'global'): Global = True

            self.advance()

            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected Identifier",
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            
            var_name = self.current_token
            self.advance()

            if self.current_token.type not in (TT_EQ, TT_DIVEQ, TT_MINEQ, TT_MULEQ, TT_PLUSEQ, TT_POWEQ):
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected '=', '-=', '+=', '*=', '/=' or '^='",
                    self.current_token.pos_start, self.current_token.pos_end
                ))
            
            if self.current_token.type in (TT_DIVEQ, TT_MINEQ, TT_MULEQ, TT_PLUSEQ, TT_POWEQ):
                BinOps = {TT_DIVEQ: TT_DIV, TT_MINEQ: TT_MIN, TT_MULEQ: TT_MUL, TT_PLUSEQ: TT_PLUS, TT_POWEQ: TT_POW}
                BinOp = Token(type=BinOps[self.current_token.type], pos_start=self.current_token.pos_start, pos_end=self.current_token.pos_end)
                self.advance()

                expr = res.register(self.expr())
                if res.error: return res

                expr = BinOpNode(VarNode(var_name), BinOp, expr)
                return res.success(VarAssignNode(var_name, expr, Global))

            self.advance()
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr, Global))

        node = res.register(self.Bin_Op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))
        if res.error: return res
        return res.success(node)

    def statement(self):
        res = ParseResult()
        token = self.current_token

        if self.current_token.matches(TT_KEYWORD, "return"):
            self.advance()
            value = None

            if self.current_token.type not in (TT_NEWLINE, TT_EOF):
                value = res.register(self.statement())
                if res.error: return res
            
            return res.success(ReturnNode(token, value))
        
        if self.current_token.matches(TT_KEYWORD, "break"):
            self.advance()
            return res.success(BreakNode(token))

        if self.current_token.matches(TT_KEYWORD, "continue"):
            self.advance()
            return res.success(ContinueNode(token))

        expr = res.register(self.expr())
        if res.error: return res

        return res.success(expr)

    def statements(self):
        res = ParseResult()
        statements = []

        while self.current_token.type == TT_NEWLINE:
            self.advance()
        
        statement = res.register(self.statement())
        if res.error: return res
        statements.append(statement)
        more_statements = True

        while self.current_token.type != TT_EOF:

            if self.current_token.type != TT_NEWLINE:
                return res.failure(InvalidSyntaxError(
                self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                "Token cannot appear after previous token", 
                self.current_token.pos_start, self.current_token.pos_end
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

            statement = res.register(self.statement())
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
        self.reset()
    
    def reset(self):
        self.value = None
        self.error = None
        self.return_value = None
        self.break_token = None
        self.continue_token = None

    def register(self, res):
        if isinstance(res, RunTimeResult):
            if res.error: self.error = res.error
            else:
                self.return_value = res.return_value
                self.break_token = res.break_token
                self.continue_token = res.continue_token
                return res.value
        return res
    
    def Return(self, value):
        self.reset()
        self.return_value = value
        return self

    def Break(self, token):
        self.reset()
        self.break_token = token
        return self

    def Continue(self, token):
        self.reset()
        self.continue_token = token
        return self

    def success(self, value):
        self.reset()
        self.value = value
        return self
    
    def failure(self, error):
        self.reset()
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
    
    def check_value(self):
        res = RunTimeResult()
        return res.success(self.token.value)

    def IllegalOperation(self, other=None, operand:str=None):
        res = RunTimeResult()
        if not other: other = self

        return res.failure(TypeError(
            self.token.pos_start.fn, self.token.pos_start.current_line,
            f"Unsupported operand type(s) for {operand}: {self.token.type.lower()} and {other.token.type.lower()}",
            self.token.pos_start, other.token.pos_end
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

    def length(self):
        res = RunTimeResult()
        return res.failure(TypeError(
            self.token.pos_start.fn, self.token.pos_start.current_line,
            f"Type {self.token.type.lower()} has no length",
            self.token.pos_start, self.token.pos_end
        ))

    def type(self):
        res = RunTimeResult()
        return res.success(String(Token(type=TT_STR, value=self.token.type.lower())))

    def __repr__(self) -> str:
        return f"{self.token.value}"

class Number(Value):
    def __init__(self, node:NumNode|Token) -> None:
        super().__init__(node)
        self.token.value = int(self.token.value) if self.token.type == TT_INT else float(self.token.value)
    
    def check_value(self):
        res = RunTimeResult()
        try: 
            return res.success(str(self.token.value))
        except: 
            return res.failure(ValueError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                "Value exceeds 4300 character limit",
                self.token.pos_start, self.token.pos_end
            ))

    def plus(self, other:Number):
        res = RunTimeResult()

        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '+')

        type=self.token.type
        value = self.token.value+other.token.value       
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def minus(self, other:Number):
        res = RunTimeResult()
        
        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '-')

        type=self.token.type        
        value = self.token.value-other.token.value               
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))

    def multiply(self, other:Number):
        res = RunTimeResult()

        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '*')

        type=self.token.type
        value = self.token.value*other.token.value
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
                self.token.pos_start, other.token.pos_end
            ))

        type=self.token.type
        value = self.token.value/other.token.value

        type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def power(self, other:Number):
        res = RunTimeResult()
        
        if other.token.type not in (TT_INT, TT_FLOAT):
            return self.IllegalOperation(other, '^')

        type=self.token.type
        value = self.token.value**other.token.value

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
        return float(self.token.value) > 0 

    def index(self, other:Number):
        res = RunTimeResult()
        return res.failure(TypeError(
            self.token.pos_start.fn, self.token.pos_start.current_line,
            f"Type {self.token.type.lower()} is not subscriptable",
            self.token.pos_start, other.token.pos_end
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
                f"Str indices must be int not '{other.token.type.lower()}'",
                self.token.pos_start, other.token.pos_end
            ))
        if int(other.token.value) >= len(self.token.value) or int(other.token.value) <= -len(self.token.value):
            return res.failure(IndexError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                "Str index out of range",
                self.token.pos_start, other.token.pos_end
            ))
        
        value = self.token.value[int(other.token.value)]
        return res.success(String(Token(type=TT_STR, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def length(self):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value=len(self.token.value), pos_start=self.token.pos_start, pos_end=self.token.pos_end)))

class List(Value):
    def __init__(self, node: ListNode|Token) -> None:
        super().__init__(node)
    
    def __iter__(self):
        i = 0
        while i < len(self.token.value):
            yield self.token.value[i]
            i+=1
    
    def plus(self, other:Value):
        res = RunTimeResult()

        value:list = self.token.value.copy()
        value.append(other)

        return res.success(List(Token(type=TT_LIST, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def multiply(self, other: Number):
        res = RunTimeResult()

        if other.token.type != TT_INT:
            return self.IllegalOperation(other, '*')

        value:list = self.token.value.copy()
        value *= other.token.value

        return res.success(List(Token(type=TT_LIST, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def power(self, other: List):
        res = RunTimeResult()

        if other.token.type != TT_LIST:
            return self.IllegalOperation(other, '^')

        value:list = self.token.value.copy()
        value.extend(other.token.value)

        return res.success(List(Token(type=TT_LIST, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def index(self, other:Number|NumNode):
        res = RunTimeResult()
        
        if other.token.type != TT_INT:
            return res.failure(TypeError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                f"List indices must be int not '{other.token.type.lower()}'",
                self.token.pos_start, other.token.pos_end
            ))
        if int(other.token.value) >= len(self.token.value) or int(other.token.value) <= -len(self.token.value):
            return res.failure(IndexError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                "List index out of range",
                self.token.pos_start, other.token.pos_end
            ))
        
        value = self.token.value[int(other.token.value)]
        return res.success(value)

    def length(self):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value=len(self.token.value), pos_start=self.token.pos_start, pos_end=self.token.pos_end)))

class Dictionary(Value):
    def __init__(self, node: DictNode|Token) -> None:
        super().__init__(node)

    def index(self, other: Value):
        res = RunTimeResult()

        if other.token.value not in self.token.value:
            return res.failure(KeyError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                f"Dictionary has no key {other.token.value}",
                self.token.pos_start, other.token.pos_end
            ))
        
        return res.success(self.token.value[other.token.value])

    def length(self):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value=len(self.token.value), pos_start=self.token.pos_start, pos_end=self.token.pos_end)))

class Class(Value):
    def __init__(self, token:Token, symbol_table:SymbolTable, init=False, parent=None) -> None:
        self.token = token
        self.symbol_table = symbol_table
        self.inizitialized = init
        self.parent = parent
    
    def Initialize(self, args:list, symbol_table):
        res = RunTimeResult()
        self.inizitialized = True

        func:Function = self.symbol_table.get_variable("init")
        if func and func.token.type == TT_FUNC:
            res.register(func.execute(args, SymbolTable(symbol_table)))
            if res.error: return res

        return res.success(Class(self.token, self.symbol_table.copy(), self.inizitialized))

#######################################
# Functions
#######################################

class Function(Value):
    def __init__(self, token:Token, func_args:list[Token], exprs:list) -> None:
        self.args = func_args
        self.exprs = exprs
        self.token = token
        
    def new_symbol_table(self, parent):
        symbol_table = SymbolTable(parent=parent)
        self.symbol_table = symbol_table

    def check_args(self, args):
        res = RunTimeResult()
        required_args = 0
        optional_args = 0

        for arg in self.args:
            if not arg[1]:
                required_args += 1
            else:
                optional_args += 1

        if len(self.args) < len(args) or required_args > len(args):
            if optional_args:
                details = f"takes from {required_args} to {len(self.args)}"
            else:
                details = f"takes {required_args}"

            return res.failure(TypeError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                f"{self.token.value} {details} argument(s) but {len(args)} were given",
                self.token.pos_start, self.token.pos_end
            ))

    def set_args(self, args):
        res = RunTimeResult()
        res.register(self.check_args(args))
        if res.error: return res

        if len(self.args) > len(args): args.extend([None]*(len(self.args)-len(args)))

        for i, var in enumerate(self.args):
            if not args[i]: arg = var[1]
            else: arg = args[i]
            self.symbol_table.assign(var[0].value, arg)
        
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        self.new_symbol_table(parent_symbol_table)
        res.register(self.set_args(args))
        if res.error: return res

        interpreter = Interpreter()
        for expr in self.exprs:
            res.register(interpreter.visit(expr, self.symbol_table))
            if res.error: return res
            if res.return_value: return res.success(res.return_value)

        return res.success(GlobalSymbolTable.get_variable("Null"))

    def __repr__(self) -> str:
        return f"{self.token.value}"

class BuiltInFunction(Function):
    def __init__(self, func_name, args:list[str]) -> None:
        self.token = Token(type=TT_FUNC, value=func_name)
        self.args = []
        for arg in args:
            self.args.append((Token(TT_IDENTIFIER, arg[0]), arg[1]))
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        self.new_symbol_table(parent_symbol_table)
        res.register(self.set_args(args))
        if res.error: return res

class Print(BuiltInFunction):
    def __init__(self) -> None: 
        super().__init__("<print>", [("value", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        variable = self.symbol_table.get_variable("value")
        if type(variable) == List:
            print([item.token.value for item in variable.token.value])
        else:
            print(variable.token.value)

        return res.success(GlobalSymbolTable.get_variable("Null"))

class Length(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<length>", [("value", None)])

    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        return self.symbol_table.get_variable("value").length()

class Type(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<type>", [("value", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        return self.symbol_table.get_variable("value").type()

class Str(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<str>", [("value", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        token = self.symbol_table.get_variable("value").token
        type = TT_STR; value = str(token.value)
        result = String(Token(type=type, value=value, pos_start=token.pos_start, pos_end=token.pos_end))
        
        return result

class Int(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<int>", [("value", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        token = self.symbol_table.get_variable("value").token
        if token.type not in (TT_INT, TT_FLOAT):
            for i in token.value:
                if type(i) != str or i not in DIGITS + '.':
                    return res.failure(ValueError(
                        token.pos_start.fn, token.pos_start.current_line,
                        "Can't convert value to int",
                        token.pos_start, token.pos_end
                    )) 
        
        value = int(float(token.value))
        type = TT_INT
        result = Number(Token(type=type, value=value, pos_start=token.pos_start, pos_end=token.pos_end))

        return result 

class Float(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<float>", [("value", None)])

    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        token = self.symbol_table.get_variable("value").token
        if token.type not in (TT_INT, TT_FLOAT):
            for i in token.value:
                if type(i) != str or i not in DIGITS + '.':
                    return res.failure(ValueError(
                        token.pos_start.fn, token.pos_start.current_line,
                        "Can't convert value to float",
                        token.pos_start, token.pos_end
                    )) 
        
        value = float(token.value)
        type = TT_FLOAT
        result = Number(Token(type=type, value=value, pos_start=token.pos_start, pos_end=token.pos_end))

        return result    

class List_(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<list>", [("value", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        token = self.symbol_table.get_variable("value").token
        if token.type not in (TT_STR, TT_LIST):
            return res.failure(ValueError(
                token.pos_start.fn, token.pos_start.current_line,
                "Can't convert value to list",
                token.pos_start, token.pos_end
            ))
        
        value = list(token.value)
        type = TT_LIST

        if token.type == TT_STR:
            value = [String(Token(type=TT_STR, value=item, pos_start=token.pos_start, pos_end=token.pos_end)) for item in value]

        result = List(Token(type=type, value=value, pos_start=token.pos_start, pos_end=token.pos_end))

        return result

class Execute(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<execute>", [("input", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        var = self.symbol_table.get_variable("input")
        _, error = Main(str(var.token.value), var.token.pos_start.fn)
        if error: return res.failure(error)

        return res.success(GlobalSymbolTable.get_variable("Null"))

class Exit(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<exit>", [])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        exit()

class Add_Element(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<add_element>", [("dict", None), ("key", None), ("value", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        dict:Dictionary = self.symbol_table.get_variable("dict")
        if dict.token.type != TT_DICT:
            return res.failure(TypeError(
                dict.token.pos_start.fn, dict.token.pos_start.current_line,
                f"Expected dict not {dict.token.type.lower()}",
                dict.token.pos_start, dict.token.pos_end
            ))
        
        key = self.symbol_table.get_variable("key")
        if key.token.type not in (TT_STR, TT_INT, TT_FLOAT): 
            return res.failure(TypeError(
                key.token.pos_start.fn, key.token.pos_start.current_line,
                f"Key can not be {key.token.type.lower()}",
                key.token.pos_start, key.token.pos_end
            ))

        value = self.symbol_table.get_variable("value")
        dict.token.value[key] = value
        dict.token.pos_end = value.token.pos_end

        return res.success(dict)

class Input(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<input>", [("prompt", String(Token(type=TT_STR, value="")))])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        prompt = self.symbol_table.get_variable("prompt")
        value = input(prompt.token.value)
        return res.success(String(Token(type=TT_STR, value=value, pos_start=prompt.token.pos_start, pos_end=prompt.token.pos_end)))

class ReadFile(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<read_file>", [("file", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        file = self.symbol_table.get_variable("file")

        if not path.isfile(file.token.value):
            return res.failure(NameError(
                file.token.pos_start.fn, file.token.pos_start.current_line,
                f"No such file or directory: '{file.token.value}'",
                file.token.pos_start, file.token.pos_end
            ))

        with open(file.token.value, 'r') as f:
            try:
                value = f.read()
            except:
                return res.failure(ValueError(
                    file.token.pos_start.fn, file.token.pos_start.current_line,
                    "Can't read the file's content",
                    file.token.pos_start, file.token.pos_end
                ))

        return res.success(String(Token(type=TT_STR, value=value)))

class WriteFile(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<write_file>", [("file", None), ("content", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        with open(self.symbol_table.get_variable("file").token.value, 'w') as f:
            f.write(self.symbol_table.get_variable("content").token.value)
        
        return res.success(GlobalSymbolTable.get_variable("Null"))
    
class ReadToFile(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<read_to>", [("fileToRead", None), ("fileToWrite", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        read_file = ReadFile()
        write_file = WriteFile()

        file = self.symbol_table.get_variable("fileToRead")
        content = res.register(read_file.execute([file], parent_symbol_table))
        
        file = self.symbol_table.get_variable("fileToWrite")
        return write_file.execute([file, content], parent_symbol_table)

class Super(BuiltInFunction):
    def __init__(self) -> None:
        super().__init__("<super>", [("class", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        Class_:Class = self.symbol_table.get_variable("class")
        if Class_.token.type != TT_CLASS:
            return res.failure(TypeError(
                Class_.token.pos_start.fn, Class_.token.pos_start.current_line,
                f"{Class_.token.type.lower()} has no parent",
                Class_.token.pos_start, Class_.token.pos_end
            ))

        if Class_.parent:
            return res.success(Class_.parent)

class Error_func(BuiltInFunction):
    def __init__(self, type:Error) -> None:
        self.type_ = type
        super().__init__("<error>", [("details", None)])
    
    def execute(self, args, parent_symbol_table):
        res = RunTimeResult()
        res.register(super().execute(args, parent_symbol_table))
        if res.error: return res

        details = self.symbol_table.get_variable("details")
        return res.failure(self.type_(
            details.token.pos_start.fn, details.token.pos_start.current_line,
            details.token.value,
            details.token.pos_start, details.token.pos_end
        ))

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
    
    def visit_ContinueNode(self, node:ContinueNode, _):
        res = RunTimeResult()
        return res.Continue(node.token)

    def visit_BreakNode(self, node:BreakNode, _):
        res = RunTimeResult()
        return res.Break(node.token)

    def visit_ReturnNode(self, node:ReturnNode, symbol_table):
        res = RunTimeResult()

        value = None
        if node.return_value:
            value = res.register(self.visit(node.return_value, symbol_table))
            if res.error: return res

        if not value:
            value = GlobalSymbolTable.get_variable("Null")
        
        value.token.pos_start = node.token.pos_start
        value.token.pos_end = node.token.pos_end
        return res.Return(value)

    def visit_DictNode(self, node:DictNode, symbol_table):
        res = RunTimeResult()
        dict_ = {}
        keys = []

        for element in node.elements:
            key = res.register(self.visit(element[0], symbol_table))
            if res.error: return res

            if key.token.type not in (TT_STR, TT_INT, TT_FLOAT): 
                return res.failure(TypeError(
                    key.token.pos_start.fn, key.token.pos_start.current_line,
                    f"Key can not be {key.token.type.lower()}",
                    key.token.pos_start, key.token.pos_end
                ))

            value = res.register(self.visit(element[1], symbol_table))
            if res.error: return res
            dict_[key.token.value] = value

        if len(keys): pos_start = keys[0].token.pos_start
        else: pos_start = None
        if len(keys) > 1: pos_end = keys[-1].token.pos_end
        else: pos_end = None

        return res.success(Dictionary(Token(type=TT_DICT, value=dict_, pos_start=pos_start, pos_end=pos_end)))

    def visit_ListNode(self, node:ListNode, symbol_table):
        res = RunTimeResult()
        results = []

        for statement in node.exprs:
            result = res.register(self.visit(statement, symbol_table))
            if res.error: return res
            
            if type(result) == RunTimeResult and result.error:
                return result

            if res.return_value:
                return res.failure(InvalidSyntaxError(
                    res.return_value.token.pos_start.fn, res.return_value.token.pos_start.current_line,
                    "Can't use 'return' outside of a function",
                    res.return_value.token.pos_start, res.return_value.token.pos_end 
                ))
            
            if res.break_token or res.continue_token:
                token = res.break_token or res.continue_token
                return res.failure(InvalidSyntaxError(
                    token.pos_start.fn, token.pos_start.current_line,
                    f"Can't use '{token.value}' outside of a loop",
                    token.pos_start, token.pos_end
                ))
            results.append(result)

        
        if len(results): pos_start = results[0].token.pos_start
        else: pos_start = None
        if len(results) > 1: pos_end = results[-1].token.pos_end
        else: pos_end = None
        
        return res.success(List(Token(type=TT_LIST, value=results, pos_start=pos_start, pos_end=pos_end)))

    def visit_ClassNode(self, node:ClassNode, symbol_table:SymbolTable):
        res = RunTimeResult()
        class_symbol_table = SymbolTable()

        parent = None
        if node.parent:
            parent = res.register(self.visit(node.parent, symbol_table))
            if res.error: return res

            if parent.token.type != TT_CLASS:
                return res.failure(TypeError(
                    parent.token.pos_start.fn, parent.token.pos_start.current_line,
                    f"Class can't inherit from {parent.token.type.lower()}",
                    parent.token.pos_start, parent.token.pos_end
                ))

            class_symbol_table = parent.symbol_table.copy()

        class_symbol_table.assign("Parent Table", symbol_table)
        for statement in node.statements:
            res.register(self.visit(statement, class_symbol_table))
            if res.error: return res
        
        symbol_table.assign(node.class_name.value, Class(node.class_name, class_symbol_table, parent=parent))
        return res.success(GlobalSymbolTable.get_variable("Null"))

    def visit_CallNode(self, node:CallNode, symbol_table):
        res = RunTimeResult()

        func:Function|Class = res.register(self.visit(node.name, symbol_table))
        if res.error: return res

        if func.token.type not in (TT_FUNC, TT_CLASS):
            return res.failure(TypeError(
                node.token.pos_start.fn, node.token.pos_start.current_line,
                f"{func.token.type.lower()} is not callable",
                node.token.pos_start, node.token.pos_end
            ))

        args = []

        for expr in node.args:
            arg = res.register(self.visit(expr, symbol_table))
            if res.error: return res
            args.append(arg)

        if func.token.type == TT_FUNC:
            if node.cls:
                cls = res.register(self.visit(node.cls, symbol_table))
                if res.error: return res
                symbol_table = cls.symbol_table.get_variable("Parent Table")
                args.insert(0, cls)

            result = res.register(func.execute(args, symbol_table))
            if res.error: return res

        elif func.token.type == TT_CLASS:
            if func.inizitialized:
                return res.failure(TypeError(
                    func.token.pos_start.fn, func.token.pos_start.current_line,
                    f"{func.token.value} is not callable",
                    func.token.pos_start, func.token.pos_end
                ))
            args.insert(0, func)

            symbol_table = func.symbol_table.get_variable("Parent Table")
            result = res.register(func.Initialize(args, symbol_table))
            if res.error: return res

        return result

    def visit_FuncDefNode(self, node:FuncDefNode, symbol_table):
        res = RunTimeResult()

        func = Function(node.token, node.args, node.exprs)
        symbol_table.assign(node.func_name.value, func)
        
        return res.success(GlobalSymbolTable.get_variable("Null"))

    def visit_TryNode(self, node:TryNode, symbol_table):
        res = RunTimeResult()
        error_found = False

        for expr in node.exprs:
            res.register(self.visit(expr, symbol_table))
            if res.error:
                error_found = True 
                res.reset()
                break
            if res.break_token or res.continue_token: return res
        
        if error_found:
            for expr in node.except_exprs:
                res.register(self.visit(expr, symbol_table))
                if res.error or res.break_token or res.continue_token: return res

        for expr in node.finally_exprs:
            res.register(self.visit(expr, symbol_table))
            if res.error or res.break_token or res.continue_token: return res
        
        return res.success(GlobalSymbolTable.get_variable("Null"))

    def visit_ForNode(self, node:ForNode, symbol_table):
        res = RunTimeResult()
        condition = res.register(self.visit(VarAssignNode(node.var_name, node.expr), symbol_table))
        if res.error: return res

        value:Value = res.register(self.visit(node.expr, symbol_table))
        if res.error: return res

        if value.token.type not in (TT_STR,TT_LIST):
            return res.failure(TypeError(
                value.token.pos_start.fn, value.token.pos_start.current_line,
                f"{value.token.type.lower()} is not iterable",
                value.token.pos_start, value.token.pos_end
            ))

        for i in value:
            if type(value) == String: condition = res.register(self.visit(VarAssignNode(node.var_name, StrNode(Token(type=TT_STR, value = i))), symbol_table))
            else: condition = res.register(self.visit(VarAssignNode(node.var_name, i), symbol_table))
            if res.error: return res
            for expr in node.exprs:
                res.register(self.visit(expr, symbol_table))
                if res.error or res.return_value: return res
                if res.continue_token or res.break_token: break
            if res.break_token: break
        
        return res.success(condition)

    def visit_WhileNode(self, node:WhileNode, symbol_table):
        res = RunTimeResult()

        condition:Number = res.register(self.visit(node.condition, symbol_table))
        if res.error: return res

        while condition.is_true():
            for expr in node.exprs:
                res.register(self.visit(expr, symbol_table))
                if res.error or res.return_value: return res
                if res.continue_token or res.break_token: break
            
            if res.break_token: break
            condition = res.register(self.visit(node.condition, symbol_table))
            if res.error: return res

        return res.success(condition)

    def visit_IfNode(self, node:IfNode, symbol_table):
        res = RunTimeResult()

        condition:Value = res.register(self.visit(node.condition, symbol_table))
        if res.error: return res

        if condition.is_true():
            for expr in node.exprs:
                expr = res.register(self.visit(expr, symbol_table))
                if res.error or res.return_value or res.break_token or res.continue_token: return res

        else:
            for expr in node.else_:
                expr = res.register(self.visit(expr, symbol_table))
                if res.error or res.return_value or res.break_token or res.continue_token: return res
        
        return res.success(condition)
    
    def visit_IndexAccessNode(self, node:IndexAccessNode, symbol_table):
        res = RunTimeResult()
        
        value:Value = res.register(self.visit(node.token, symbol_table))
        if res.error: return res
        index:Number = res.register(self.visit(node.index, symbol_table)) 
        if res.error: return res

        char = res.register(value.index(index))
        return res.success(char)

    def visit_AttributeNode(self, node:AttributeNode, symbol_table):
        res = RunTimeResult()

        class_:Class = res.register(self.visit(node.class_name, symbol_table))
        if res.error: return res
        
        value = class_.symbol_table.get_variable(node.attribute.value)

        if not value:
            return res.failure(AttributeError_(
                class_.token.pos_start.fn, class_.token.pos_start.current_line,
                f"{class_.token.value} has no attribute '{node.attribute.value}'",
                class_.token.pos_start, class_.token.pos_end
            ))
        
        return res.success(value)

    def visit_AttributeAssignNode(self, node:AttributeAssignNode, symbol_table):
        res = RunTimeResult()

        class_:Class = res.register(self.visit(node.class_name, symbol_table))
        if res.error: return res

        if class_.token.type != TT_CLASS:
            return res.failure(AttributeError_(
                class_.token.pos_start.fn, class_.token.pos_start.current_line,
                f"{class_.token.type.lower()} has no attribute '{node.attribute.value}'",
                class_.token.pos_start, class_.token.pos_end
            ))

        value = res.register(self.visit(node.value, symbol_table))
        if res.error: return res

        class_.symbol_table.assign(node.attribute.value, value)

        return res.success(GlobalSymbolTable.get_variable("Null"))

    def visit_VarNode(self, node:VarNode, symbol_table):
        res = RunTimeResult()
        var_name = node.var.value
        value = symbol_table.get_variable(var_name)

        if not value:
            return res.failure(NameError(
                node.var.pos_start.fn, node.var.pos_start.current_line,
                f"{var_name} is not defined",
                node.var.pos_start, node.var.pos_end
            ))
        
        value.token.pos_start = node.var.pos_start
        value.token.pos_end = node.var.pos_end
        return res.success(value)
    
    def visit_VarAssignNode(self, node:VarAssignNode, symbol_table):
        res = RunTimeResult()
        var_name = node.var_name.value
        value = res.register(self.visit(node.value, symbol_table))
        if res.error: return res

        if node.Global: symbol_table = GlobalSymbolTable

        symbol_table.assign(var_name, value)
        return res.success(value)

    def visit_StrNode(self, node:StrNode, _):
        res = RunTimeResult()
        node = String(node)
        return res.success(node)

    def visit_NumNode(self, node:NumNode, _):
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
        
        res.register(result.check_value())
        if res.error: return res

        return res.success(result)

GlobalSymbolTable = SymbolTable()
GlobalSymbolTable.add_keywords(
    'var', 'global', 'def', 'class',
    'try', 'except', 'finally',
    'return', 'break', 'continue',
    'and', 'or', 'not', 
    'if', 'then', 'else', 'fi', 
    'while', 'for', 'in', 'end'
)
GlobalSymbolTable.assign_multiple(
    ("Null", Number(Token(type=TT_INT, value=0))), 
    ("True", Number(Token(type=TT_INT, value=1))),
    ("False", Number(Token(type=TT_INT, value=0))),
    ("PI", Number(Token(type=TT_FLOAT, value=str(math.pi)))),
    ("print", Print()),
    ("length", Length()),
    ("type", Type()),
    ("str", Str()),
    ("int", Int()),
    ("float", Float()),
    ("list", List_()),
    ("execute", Execute()),
    ("exit", Exit()),
    ("add_element", Add_Element()),
    ("input", Input()),
    ("read_file", ReadFile()),
    ("write_file", WriteFile()),
    ("read_to", ReadToFile()),
    ("super", Super()),
    ("_all_", Dictionary(Token(type= TT_DICT, value=GlobalSymbolTable.symbols))),
    ("TypeError", Error_func(TypeError)), ("KeyError", Error_func(KeyError)), ("AttributeError", Error_func(AttributeError_)),
    ("IndexError", Error_func(IndexError)), ("NameError", Error_func(NameError)), ("ValueError", Error_func(ValueError)),
    ("ZeroDivisionError", Error_func(ZeroDivisionError)), ("InvalidSyntaxError", Error_func(InvalidSyntaxError)), ("IllegalCharError", Error_func(IllegalCharError))
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