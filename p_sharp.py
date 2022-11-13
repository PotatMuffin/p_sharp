from __future__ import annotations
from arrows import add_arrows
import string

#############################
# constants
#############################

DIGITS = string.digits
LETTERS = f"{string.ascii_letters}_"

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

#############################
#Symbol Table
#############################

class SymbolTable:
    def __init__(self) -> None:
        self.symbols = {}
        self.keywords = []
        self.parent:SymbolTable = None
    
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
            self.idx = 0
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
TT_LPAREN = "LPAREN"
TT_RPAREN = "RPAREN"

TT_INT = "INT"
TT_FLOAT = "FLOAT"
TT_BOOL = "BOOL"

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
        return f"{self.type}" if not self.value else f"{self.type}:{self.value}" 

    def matches(self, type, value):
        return self.type == type and self.value == value

#############################
# Lexer
#############################

class Lexer:
    def __init__(self, input, fn) -> None:
        self.pos = Position(0, 1, input, fn)
    
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

    
    def make_tokens(self):
        tokens = []
        
        chars = {
        '+': lambda: tokens.append(Token(type=TT_PLUS, pos_start=self.pos.copy())), 
        '-': lambda: tokens.append(Token(type=TT_MIN, pos_start=self.pos.copy())), 
        '*': lambda: tokens.append(Token(type=TT_MUL, pos_start=self.pos.copy())), 
        '/': lambda: tokens.append(Token(type=TT_DIV, pos_start=self.pos.copy())),
        '(': lambda: tokens.append(Token(type=TT_LPAREN, pos_start=self.pos.copy())),
        ')': lambda: tokens.append(Token(type=TT_RPAREN, pos_start=self.pos.copy())),
        '^': lambda: tokens.append(Token(type=TT_POW, pos_start=self.pos.copy())),
        ';': lambda: tokens.append(Token(type=TT_NEWLINE, pos_start=self.pos.copy())),
        '!': lambda: tokens.append(self.make_not_equals()),
        '=': lambda: tokens.append(self.make_equals()),
        '>': lambda: tokens.append(self.make_greater_than()),
        '<': lambda: tokens.append(self.make_lower_than())
        }

        while self.pos.current_char:
            if self.pos.current_char in ' \t':
                self.pos.advance()
                continue
            if self.pos.current_char in DIGITS:
                tokens.append(self.make_digit())
                continue
            if self.pos.current_char in LETTERS:
                tokens.append(self.make_identifier())
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

class IfNode:
    def __init__(self, condition) -> None:
        self.condition = condition

    def __repr__(self) -> str:
        return f"if {self.condition}"

class ListNode:
    def __init__(self, statements:list) -> None:
        self.statements = statements
    
    def __repr__(self) -> str:
        return f"{self.statements}"
        
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
                self.current_token.pos_start.fn, self.current_token.pos_start.text,
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
    
    def if_expr(self):
        res = ParseResult()

        self.advance()
        condition = res.register(self.expr())
        if res.error: return res
    
        return res.success(IfNode(condition))

    def atom(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_INT, TT_FLOAT):
            res.register(self.advance())
            return res.success(NumNode(token))
        
        if token.type == TT_IDENTIFIER:
            res.register(self.advance())
            return res.success(VarNode(token))
        
        if token.type == TT_LPAREN:
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            if self.current_token.type == TT_RPAREN:
                res.register(self.advance())
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

        return res.failure(InvalidSyntaxError(
            self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
            "Expected int, float, '+', '-', Identifier or '('",
            self.current_token.pos_start.idx, self.current_token.pos_end.idx
        ))

    def power(self):
        return self.Bin_Op(self.atom, (TT_POW, ), self.factor)

    def factor(self):
        res = ParseResult()
        token = self.current_token

        if token.type in (TT_PLUS, TT_MIN):
            res.register(self.advance())
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
            res.register(self.advance())

            node = res.register(self.comp_expr())
            if res.error: return res
            return res.success(UnaryOpNode(token, node))
        
        node = res.register(self.Bin_Op(self.arith_expr, (TT_IS, TT_ISNOT, TT_LT, TT_GT, TT_LTE, TT_GTE)))
    
        if res.error: return res
        return res.success(node)

    def expr(self):
        res = ParseResult()
        if self.current_token.matches(TT_KEYWORD, 'var'):
            res.register(self.advance())

            if self.current_token.type != TT_IDENTIFIER:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected Identifier",
                    self.current_token.pos_start.idx, self.current_token.pos_end.idx
                ))
            
            var_name = self.current_token
            res.register(self.advance())

            if self.current_token.type != TT_EQ:
                return res.failure(InvalidSyntaxError(
                    self.current_token.pos_start.fn, self.current_token.pos_start.current_line,
                    "Expected '='",
                    self.current_token.pos_start.idx, self.current_token.pos_end.idx
                ))
            
            res.register(self.advance())
            expr = res.register(self.expr())
            if res.error: return res
            return res.success(VarAssignNode(var_name, expr))

        node = res.register(self.Bin_Op(self.comp_expr, ((TT_KEYWORD, 'and'), (TT_KEYWORD, 'or'))))
        if res.error: return res
        return res.success(node)

    def Bin_Op(self, function, operators, function2=None):
        res = ParseResult()
        if not function2:
            function2 = function

        left = res.register(function())
        if res.error: return res
        
        while self.current_token.type in operators or (self.current_token.type, self.current_token.value) in operators:
            token = self.current_token
            res.register(self.advance())
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

class Number:
    def __init__(self, node:NumNode|Token) -> None:
        if type(node) == Token:
            self.token = node
        else:
            self.token = node.token
    
    def plus(self, other:NumNode|Number):
        res = RunTimeResult()

        type=self.token.type
        value = str(eval(f"{self.token.value}+{other.token.value}"))        
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def minus(self, other:NumNode|Number):
        res = RunTimeResult()

        type=self.token.type        
        value = str(eval(f"{self.token.value}-{other.token.value}"))                
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))

    def multiply(self, other:NumNode|Number):
        res = RunTimeResult()

        type=self.token.type
        value = str(eval(f"{self.token.value}*{other.token.value}"))
        if other.token.type == TT_FLOAT:
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def divide(self, other:NumNode|Number):
        res = RunTimeResult()

        if other.token.value == "0":
            return res.failure(ZeroDivisionError(
                self.token.pos_start.fn, self.token.pos_start.current_line,
                "Can't divide by zero", 
                self.token.pos_start.idx, other.token.pos_end.idx
            ))

        type=self.token.type
        value = str(eval(f"{self.token.value}/{other.token.value}"))

        type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))
    
    def power(self, other:NumNode|Number):
        res = RunTimeResult()

        type=self.token.type
        try: value = str(eval(f"{self.token.value}**{other.token.value}"))
        except: return res.failure(ValueError(
            self.token.pos_start.fn, self.token.pos_start.current_line, 
            "Number exceeds character limit (4300)",
            self.token.pos_start.idx, other.token.pos_end.idx
        ))

        if other.token.type == TT_FLOAT or other.token.value.startswith("-"):
            type = TT_FLOAT
        return res.success(Number(NumNode(Token(type=type, value=value, pos_start=self.token.pos_start, pos_end=other.token.pos_end))))

    def equals(self, other:NumNode|Number):
        res = RunTimeResult()
        if self.token.matches(other.token.type, other.token.value):
            return res.success(Number(Token(type=TT_INT, value="1", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))
        return res.success(Number(Token(type=TT_INT, value="0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def not_equals(self, other:NumNode|Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if not self.token.matches(other.token.type, other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def greater_than(self, other:NumNode|Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if int(self.token.value) > int(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def greater_equals(self, other:NumNode|Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if int(self.token.value) >= int(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))
    
    def lower_than(self, other:NumNode|Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if int(self.token.value) < int(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def lower_equals(self, other:NumNode|Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if int(self.token.value) <= int(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def and_(self, other:NumNode|Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if int(self.token.value) and int(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def or_(self, other: NumNode|Number):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if int(self.token.value) or int(other.token.value) else "0", pos_start=self.token.pos_start, pos_end=other.token.pos_end)))

    def not_(self):
        res = RunTimeResult()
        return res.success(Number(Token(type=TT_INT, value="1" if int(self.token.value) <= 0 else "0", pos_start=self.token.pos_start, pos_end=self.token.pos_end)))

    def is_true(self):
        res = RunTimeResult()
        return int(self.token.value) > 0 

    def __repr__(self) -> str:
        return f"{self.token}"

#######################################
# Interpreter
#######################################

class Interpreter:
    def no_visit_method(self, node):
        raise AttributeError(f"no visit_{type(node).__name__} method found")

    def visit(self, node):
        method_name = f"visit_{type(node).__name__}"
        method = getattr(self, method_name, self.no_visit_method)

        return method(node)

    def visit_ListNode(self, node:ListNode):
        res = RunTimeResult()
        results = []

        for statement in node.statements:
            result = res.register(self.visit(statement))
            if res.error: return res
            results.append(result)
        
        return res.success(results)

    def visit_VarNode(self, node:VarNode):
        res = RunTimeResult()
        var_name = node.var.value
        value = GlobalSymbolTable.get_variable(var_name)

        if not value:
            return res.failure(NameError(
                node.var.pos_start.fn, node.var.pos_start.current_line,
                f"{var_name} is not defined",
                node.var.pos_start.idx, node.var.pos_end.idx
            ))
        
        value.token.pos_start = node.var.pos_start
        value.token.pos_end = node.var.pos_end
        return res.success(value)
    
    def visit_VarAssignNode(self, node:VarAssignNode):
        res = RunTimeResult()
        var_name = node.var_name.value
        value = res.register(self.visit(node.value))
        if res.error: return res

        GlobalSymbolTable.assign(var_name, value)
        return res.success(value)

    def visit_NumNode(self, node:NumNode):
        res = RunTimeResult()
        node = Number(node)
        
        if node.token.type == TT_INT:
            if len(node.token.value) > 1 and node.token.value[0] == "0":
                return res.failure(InvalidSyntaxError(
                    node.token.pos_start.fn, node.token.pos_start.current_line,
                    "Leading zeros in integers are not permitted",
                    node.token.pos_start.idx, node.token.pos_start.idx
                ))

        return res.success(node)
    
    def visit_UnaryOpNode(self, node:UnaryOpNode):
        res = RunTimeResult()

        n:Number = res.register(self.visit(node.node))
        if res.error: return res
        pos_start = n.token.pos_start
        pos_start.back()

        if node.op_token.type == TT_MIN:
            return n.multiply(Number(Token(type=TT_INT, value="-1", pos_start=pos_start)))
        elif node.op_token.type == TT_PLUS:
            return n.multiply(Number(Token(type=TT_INT, value="1", pos_start=pos_start)))
        elif node.op_token.matches(TT_KEYWORD, 'not'):
            return n.not_()

    def visit_BinOpNode(self, node:BinOpNode):
        res = RunTimeResult()

        left:Number = res.register(self.visit(node.left))
        if res.error: return res
        right:Number = res.register(self.visit(node.right))
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
GlobalSymbolTable.add_keywords('var', 'and', 'or', 'not', 'if', 'then')
GlobalSymbolTable.assign("Null", Number(Token(type=TT_INT, value="0")))

def Main(input, fn):
    lexer = Lexer(input, fn)
    tokens, error = lexer.make_tokens()
    if error: return tokens, error
    if len(tokens) == 1: return [], None

    parser = Parser(tokens)
    ast = parser.parse()
    if ast.error: return None, ast.error

    interpreter = Interpreter()
    result = interpreter.visit(ast.node) 

    return result.value, result.error