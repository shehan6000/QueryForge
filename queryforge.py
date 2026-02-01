#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║                        Q U E R Y F O R G E                         ║
║            A Database Engine — Built From Scratch                   ║
║                                                                      ║
║  Layers:  Source → Lexer → Parser → AST → Planner → Executor        ║
║  Storage: In-memory row store + B-Tree indexes + WAL transactions    ║
║  Zero external dependencies.  Python 3.10+                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import re, sys, math, copy, time, os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Any, Optional


# ════════════════════════════════════════════════════════════════════════════
# 1.  TOKEN  &  LEXER
# ════════════════════════════════════════════════════════════════════════════

class TT(Enum):
    """Token types."""
    # literals
    INT = auto(); FLOAT = auto(); STR = auto()
    TRUE = auto(); FALSE = auto(); NULL = auto()
    # identifier
    IDENT = auto()
    # keywords (mapped from lowercase text)
    CREATE = auto(); TABLE = auto(); DROP = auto()
    INSERT = auto(); INTO = auto(); VALUES = auto()
    SELECT = auto(); FROM = auto(); WHERE = auto()
    AND = auto(); OR = auto(); NOT = auto()
    UPDATE = auto(); SET = auto(); DELETE = auto()
    ORDER = auto(); BY = auto(); ASC = auto(); DESC = auto()
    LIMIT = auto(); OFFSET = auto()
    JOIN = auto(); INNER = auto(); LEFT = auto(); RIGHT = auto(); ON = auto()
    GROUP = auto(); HAVING = auto()
    AS = auto(); DISTINCT = auto()
    COUNT = auto(); SUM = auto(); AVG = auto(); MIN = auto(); MAX = auto()
    INDEX = auto(); UNIQUE = auto()
    ALTER = auto(); ADD = auto(); COLUMN = auto(); RENAME = auto(); TO = auto()
    BEGIN = auto(); COMMIT = auto(); ROLLBACK = auto()
    LIKE = auto(); IN = auto(); BETWEEN = auto(); IS = auto()
    CASE = auto(); WHEN = auto(); THEN = auto(); ELSE = auto(); END = auto()
    IF = auto(); EXISTS = auto()
    PRIMARY = auto(); KEY = auto(); FOREIGN = auto(); REFERENCES = auto()
    # type keywords
    T_INT = auto(); T_BIGINT = auto(); T_REAL = auto()
    T_TEXT = auto(); T_BLOB = auto(); T_BOOL = auto()
    # operators
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto(); PERCENT = auto()
    EQ = auto(); NEQ = auto(); LT = auto(); GT = auto(); LTE = auto(); GTE = auto()
    LPAREN = auto(); RPAREN = auto()
    COMMA = auto(); SEMI = auto(); DOT = auto(); DCOLON = auto()
    # end
    EOF = auto()


@dataclass
class Token:
    type: TT
    value: Any        # raw value for INT/FLOAT/STR/IDENT, else None
    line: int
    col:  int


KEYWORDS: dict[str, TT] = {
    "create": TT.CREATE, "table": TT.TABLE, "drop": TT.DROP,
    "insert": TT.INSERT, "into": TT.INTO, "values": TT.VALUES,
    "select": TT.SELECT, "from": TT.FROM, "where": TT.WHERE,
    "and": TT.AND, "or": TT.OR, "not": TT.NOT,
    "update": TT.UPDATE, "set": TT.SET, "delete": TT.DELETE,
    "order": TT.ORDER, "by": TT.BY, "asc": TT.ASC, "desc": TT.DESC,
    "limit": TT.LIMIT, "offset": TT.OFFSET,
    "join": TT.JOIN, "inner": TT.INNER, "left": TT.LEFT,
    "right": TT.RIGHT, "on": TT.ON,
    "group": TT.GROUP, "having": TT.HAVING,
    "as": TT.AS, "distinct": TT.DISTINCT,
    "count": TT.COUNT, "sum": TT.SUM, "avg": TT.AVG,
    "min": TT.MIN, "max": TT.MAX,
    "index": TT.INDEX, "unique": TT.UNIQUE,
    "alter": TT.ALTER, "add": TT.ADD, "column": TT.COLUMN,
    "rename": TT.RENAME, "to": TT.TO,
    "begin": TT.BEGIN, "commit": TT.COMMIT, "rollback": TT.ROLLBACK,
    "like": TT.LIKE, "in": TT.IN, "between": TT.BETWEEN, "is": TT.IS,
    "case": TT.CASE, "when": TT.WHEN, "then": TT.THEN,
    "else": TT.ELSE, "end": TT.END,
    "if": TT.IF, "exists": TT.EXISTS,
    "primary": TT.PRIMARY, "key": TT.KEY,
    "foreign": TT.FOREIGN, "references": TT.REFERENCES,
    "int": TT.T_INT, "bigint": TT.T_BIGINT, "real": TT.T_REAL,
    "text": TT.T_TEXT, "blob": TT.T_BLOB, "bool": TT.T_BOOL,
    "true": TT.TRUE, "false": TT.FALSE, "null": TT.NULL,
}


class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"[Lex {line}:{col}] {msg}")


class Lexer:
    """Hand-written single-pass lexer.  O(n), no regex on the hot path."""

    def __init__(self, src: str):
        self.src  = src
        self.pos  = 0
        self.line = 1
        self.col  = 1

    # ── public ──────────────────────────────────────────────────────────
    def tokenise(self) -> list[Token]:
        out: list[Token] = []
        while True:
            t = self._next()
            out.append(t)
            if t.type is TT.EOF:
                break
        return out

    # ── internals ───────────────────────────────────────────────────────
    def _peek(self) -> str | None:
        return self.src[self.pos] if self.pos < len(self.src) else None

    def _peek2(self) -> str | None:
        p = self.pos + 1
        return self.src[p] if p < len(self.src) else None

    def _advance(self) -> str | None:
        if self.pos >= len(self.src):
            return None
        ch = self.src[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1; self.col = 1
        else:
            self.col += 1
        return ch

    def _skip(self):
        """Skip whitespace and comments (-- and /* */)."""
        while self.pos < len(self.src):
            # whitespace
            if self.src[self.pos].isspace():
                self._advance(); continue
            # line comment
            if self.src[self.pos:self.pos+2] == '--':
                while self._advance() not in (None, '\n'):
                    pass
                continue
            # block comment
            if self.src[self.pos:self.pos+2] == '/*':
                self._advance(); self._advance()
                while self.pos < len(self.src):
                    if self.src[self.pos:self.pos+2] == '*/':
                        self._advance(); self._advance(); break
                    self._advance()
                continue
            break

    def _next(self) -> Token:
        self._skip()
        line, col = self.line, self.col
        ch = self._peek()
        if ch is None:
            return Token(TT.EOF, None, line, col)

        # ── two-char operators ──────────────────────────────────────────
        two = self.src[self.pos:self.pos+2]
        if two == '!=':  self._advance(); self._advance(); return Token(TT.NEQ, None, line, col)
        if two == '<=':  self._advance(); self._advance(); return Token(TT.LTE, None, line, col)
        if two == '>=':  self._advance(); self._advance(); return Token(TT.GTE, None, line, col)
        if two == '::':  self._advance(); self._advance(); return Token(TT.DCOLON, None, line, col)

        # ── single-char operators ───────────────────────────────────────
        singles = {
            '+': TT.PLUS, '*': TT.STAR, '/': TT.SLASH, '%': TT.PERCENT,
            '(': TT.LPAREN, ')': TT.RPAREN, ',': TT.COMMA,
            ';': TT.SEMI, '.': TT.DOT, '=': TT.EQ, '<': TT.LT, '>': TT.GT,
        }
        if ch in singles:
            self._advance()
            return Token(singles[ch], None, line, col)

        # minus — could be operator or start of negative number
        if ch == '-':
            self._advance()
            if self._peek() and self._peek().isdigit():
                return self._lex_number(line, col, neg=True)
            return Token(TT.MINUS, None, line, col)

        # ── string literal ──────────────────────────────────────────────
        if ch == "'":
            return self._lex_string(line, col)

        # ── number ──────────────────────────────────────────────────────
        if ch.isdigit():
            return self._lex_number(line, col, neg=False)

        # ── identifier / keyword ────────────────────────────────────────
        if ch.isalpha() or ch == '_':
            return self._lex_ident(line, col)

        raise LexError(f"unexpected character '{ch}'", line, col)

    # ── number ──────────────────────────────────────────────────────────
    def _lex_number(self, line: int, col: int, neg: bool) -> Token:
        buf = '-' if neg else ''
        while self._peek() and self._peek().isdigit():
            buf += self._advance()
        is_float = False
        # decimal part
        if self._peek() == '.' and self._peek2() and self._peek2().isdigit():
            is_float = True
            buf += self._advance()  # '.'
            while self._peek() and self._peek().isdigit():
                buf += self._advance()
        # exponent
        if self._peek() in ('e', 'E'):
            is_float = True
            buf += self._advance()
            if self._peek() in ('+', '-'):
                buf += self._advance()
            while self._peek() and self._peek().isdigit():
                buf += self._advance()
        if is_float:
            return Token(TT.FLOAT, float(buf), line, col)
        return Token(TT.INT, int(buf), line, col)

    # ── string ──────────────────────────────────────────────────────────
    def _lex_string(self, line: int, col: int) -> Token:
        self._advance()  # opening '
        buf = ''
        while True:
            ch = self._advance()
            if ch is None:
                raise LexError("unterminated string", line, col)
            if ch == "'":
                if self._peek() == "'":       # escaped ''
                    self._advance()
                    buf += "'"
                else:
                    break
            else:
                buf += ch
        return Token(TT.STR, buf, line, col)

    # ── ident ───────────────────────────────────────────────────────────
    def _lex_ident(self, line: int, col: int) -> Token:
        buf = ''
        while self._peek() and (self._peek().isalnum() or self._peek() == '_'):
            buf += self._advance()
        low = buf.lower()
        if low in KEYWORDS:
            return Token(KEYWORDS[low], buf, line, col)
        return Token(TT.IDENT, buf, line, col)


# ════════════════════════════════════════════════════════════════════════════
# 2.  AST  (Abstract Syntax Tree nodes)
# ════════════════════════════════════════════════════════════════════════════

# ── Expressions ─────────────────────────────────────────────────────────────
@dataclass
class Expr:
    """Base — never instantiated directly."""

@dataclass
class LitInt(Expr):    val: int
@dataclass
class LitFloat(Expr):  val: float
@dataclass
class LitStr(Expr):    val: str
@dataclass
class LitBool(Expr):   val: bool
@dataclass
class LitNull(Expr):   pass

@dataclass
class ColRef(Expr):
    table: str | None
    name:  str

@dataclass
class BinOp(Expr):
    op:    str          # '+' '-' '*' '/' '%' '=' '!=' '<' '>' '<=' '>=' 'AND' 'OR'
    left:  Expr
    right: Expr

@dataclass
class UnaryNot(Expr):  expr: Expr
@dataclass
class UnaryMinus(Expr): expr: Expr

@dataclass
class FuncCall(Expr):
    name:     str       # COUNT SUM AVG MIN MAX or user-defined
    args:     list[Expr]
    distinct: bool

@dataclass
class CaseExpr(Expr):
    operand: Expr | None                  # simple CASE x WHEN …
    whens:   list[tuple[Expr, Expr]]      # (cond, result)
    else_:   Expr | None

@dataclass
class IsNullExpr(Expr):
    expr:   Expr
    negated: bool   # IS NOT NULL

@dataclass
class LikeExpr(Expr):
    expr:    Expr
    pattern: Expr
    negated: bool

@dataclass
class InExpr(Expr):
    expr:    Expr
    values:  list[Expr] | None    # literal list
    subsel:  SelectStmt | None    # or sub-select
    negated: bool

@dataclass
class BetweenExpr(Expr):
    expr:    Expr
    low:     Expr
    high:    Expr
    negated: bool

@dataclass
class ExistsExpr(Expr):
    subsel: SelectStmt

@dataclass
class SubSelectExpr(Expr):
    subsel: SelectStmt

@dataclass
class CastExpr(Expr):
    expr: Expr
    to:   str

@dataclass
class StarExpr(Expr):   pass   # bare *

# ── Statements ──────────────────────────────────────────────────────────────
@dataclass
class ColumnDef:
    name:       str
    col_type:   str     # INT BIGINT REAL TEXT BLOB BOOL
    not_null:   bool = False
    unique:     bool = False
    primary_key:bool = False
    default:    Expr | None = None

@dataclass
class CreateTableStmt:
    name:           str
    columns:        list[ColumnDef]
    if_not_exists:  bool = False
    primary_key:    list[str] | None = None
    foreign_keys:   list[dict] = field(default_factory=list)

@dataclass
class DropTableStmt:
    name:      str
    if_exists: bool = False

@dataclass
class InsertStmt:
    table:   str
    columns: list[str] | None   # None means all, in order
    rows:    list[list[Expr]]

# SELECT pieces
@dataclass
class SelectCol:
    expr:  Expr
    alias: str | None = None

@dataclass
class FromTable:
    name:  str
    alias: str | None = None

@dataclass
class JoinClause:
    left:      Any   # FromTable | JoinClause
    join_type: str   # INNER LEFT RIGHT
    right:     FromTable
    on:        Expr

@dataclass
class OrderItem:
    expr: Expr
    asc:  bool = True

@dataclass
class SelectStmt:
    columns:  list[SelectCol | StarExpr]
    from_:    FromTable | JoinClause | None = None
    where_:   Expr | None = None
    group_by: list[Expr] = field(default_factory=list)
    having:   Expr | None = None
    order_by: list[OrderItem] = field(default_factory=list)
    limit:    Expr | None = None
    offset:   Expr | None = None
    distinct: bool = False

@dataclass
class UpdateStmt:
    table:  str
    sets:   list[tuple[str, Expr]]
    where_: Expr | None = None

@dataclass
class DeleteStmt:
    table:  str
    where_: Expr | None = None

@dataclass
class CreateIndexStmt:
    name:    str
    table:   str
    columns: list[str]
    unique:  bool = False

@dataclass
class DropIndexStmt:
    name:      str
    if_exists: bool = False

@dataclass
class AlterAddCol(object):
    table: str
    col:   ColumnDef

@dataclass
class AlterDropCol(object):
    table: str
    col:   str

@dataclass
class AlterRenameCol(object):
    table:  str
    old:    str
    new:    str

@dataclass
class AlterRenameTable(object):
    old: str
    new: str

class BeginStmt:    pass
class CommitStmt:   pass
class RollbackStmt: pass


# ════════════════════════════════════════════════════════════════════════════
# 3.  PARSER  (recursive descent)
# ════════════════════════════════════════════════════════════════════════════

class ParseError(Exception):
    def __init__(self, msg: str, tok: Token):
        super().__init__(f"[Parse {tok.line}:{tok.col}] {msg}  (got {tok.type.name}"
                         + (f" '{tok.value}'" if tok.value is not None else "") + ")")


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos    = 0

    # ── helpers ───────────────────────────────────────────────────────
    def _cur(self) -> Token:   return self.tokens[self.pos]
    def _peek(self) -> TT:     return self._cur().type
    def _advance(self) -> Token:
        t = self._cur()
        if self.pos < len(self.tokens) - 1:
            self.pos += 1
        return t

    def _eat(self, tt: TT) -> Token:
        if self._peek() != tt:
            raise ParseError(f"expected {tt.name}", self._cur())
        return self._advance()

    def _eat_if(self, tt: TT) -> bool:
        if self._peek() == tt:
            self._advance(); return True
        return False

    def _eat_ident(self) -> str:
        # Accept IDENT *or* any keyword used as identifier (e.g. column named 'key')
        t = self._cur()
        if t.type == TT.IDENT:
            self._advance()
            return t.value
        # Allow reserved words as identifiers in column/table name positions
        if t.value and isinstance(t.value, str) and t.type in KEYWORD_AS_IDENT:
            self._advance()
            return t.value
        raise ParseError("expected identifier", t)

    def _ident_list(self) -> list[str]:
        out = [self._eat_ident()]
        while self._eat_if(TT.COMMA):
            out.append(self._eat_ident())
        return out

    # ─── top ──────────────────────────────────────────────────────────
    def parse(self) -> Any:
        stmt = self._statement()
        self._eat_if(TT.SEMI)   # optional trailing semicolon
        return stmt

    def _statement(self):
        match self._peek():
            case TT.CREATE:   return self._create()
            case TT.DROP:     return self._drop()
            case TT.INSERT:   return self._insert()
            case TT.SELECT:   return self._select()
            case TT.UPDATE:   return self._update()
            case TT.DELETE:   return self._delete()
            case TT.ALTER:    return self._alter()
            case TT.BEGIN:    self._advance(); return BeginStmt()
            case TT.COMMIT:   self._advance(); return CommitStmt()
            case TT.ROLLBACK: self._advance(); return RollbackStmt()
            case _: raise ParseError("expected statement", self._cur())

    # ─── CREATE ───────────────────────────────────────────────────────
    def _create(self):
        self._eat(TT.CREATE)
        if self._peek() == TT.TABLE:
            return self._create_table()
        if self._peek() in (TT.INDEX, TT.UNIQUE):
            return self._create_index()
        raise ParseError("expected TABLE or INDEX", self._cur())

    def _create_table(self):
        self._eat(TT.TABLE)
        ine = False
        if self._peek() == TT.IF:
            self._eat(TT.IF); self._eat(TT.NOT); self._eat(TT.EXISTS)
            ine = True
        name = self._eat_ident()
        self._eat(TT.LPAREN)

        cols, pk, fks = [], None, []
        while True:
            if self._peek() == TT.PRIMARY:
                self._eat(TT.PRIMARY); self._eat(TT.KEY)
                self._eat(TT.LPAREN)
                pk = self._ident_list()
                self._eat(TT.RParen if False else TT.RPAREN)
            elif self._peek() == TT.FOREIGN:
                self._eat(TT.FOREIGN); self._eat(TT.KEY)
                self._eat(TT.LPAREN)
                fk_cols = self._ident_list()
                self._eat(TT.RPAREN)
                self._eat(TT.REFERENCES)
                ref_tbl = self._eat_ident()
                self._eat(TT.LPAREN)
                ref_cols = self._ident_list()
                self._eat(TT.RPAREN)
                fks.append({"columns": fk_cols, "ref_table": ref_tbl, "ref_cols": ref_cols})
            else:
                cols.append(self._column_def())
            if not self._eat_if(TT.COMMA):
                break
        self._eat(TT.RPAREN)
        return CreateTableStmt(name=name, columns=cols, if_not_exists=ine,
                               primary_key=pk, foreign_keys=fks)

    def _column_def(self) -> ColumnDef:
        name = self._eat_ident()
        TYPE_MAP = {TT.T_INT: "INT", TT.T_BIGINT: "BIGINT", TT.T_REAL: "REAL",
                    TT.T_TEXT: "TEXT", TT.T_BLOB: "BLOB", TT.T_BOOL: "BOOL"}
        if self._peek() not in TYPE_MAP:
            raise ParseError("expected type (INT/BIGINT/REAL/TEXT/BLOB/BOOL)", self._cur())
        col_type = TYPE_MAP[self._advance().type]
        pk = nn = uniq = False
        default = None
        # constraint flags (any order)
        while True:
            if self._peek() == TT.PRIMARY:
                self._eat(TT.PRIMARY); self._eat(TT.KEY); pk = True
            elif self._peek() == TT.NOT:
                self._eat(TT.NOT); self._eat(TT.NULL); nn = True
            elif self._peek() == TT.UNIQUE:
                self._eat(TT.UNIQUE); uniq = True
            else:
                break
        return ColumnDef(name=name, col_type=col_type, not_null=nn,
                         unique=uniq, primary_key=pk, default=default)

    # ─── DROP ─────────────────────────────────────────────────────────
    def _drop(self):
        self._eat(TT.DROP)
        if self._peek() == TT.TABLE:
            self._eat(TT.TABLE)
            ie = False
            if self._peek() == TT.IF:
                self._eat(TT.IF); self._eat(TT.EXISTS); ie = True
            return DropTableStmt(name=self._eat_ident(), if_exists=ie)
        if self._peek() == TT.INDEX:
            self._eat(TT.INDEX)
            ie = False
            if self._peek() == TT.IF:
                self._eat(TT.IF); self._eat(TT.EXISTS); ie = True
            return DropIndexStmt(name=self._eat_ident(), if_exists=ie)
        raise ParseError("expected TABLE or INDEX after DROP", self._cur())

    # ─── INSERT ───────────────────────────────────────────────────────
    def _insert(self):
        self._eat(TT.INSERT); self._eat(TT.INTO)
        table = self._eat_ident()
        cols = None
        if self._eat_if(TT.LPAREN):
            cols = self._ident_list()
            self._eat(TT.RPAREN)
        self._eat(TT.VALUES)
        rows = []
        while True:
            self._eat(TT.LPAREN)
            row = [self._expr()]
            while self._eat_if(TT.COMMA):
                row.append(self._expr())
            self._eat(TT.RPAREN)
            rows.append(row)
            if not self._eat_if(TT.COMMA):
                break
        return InsertStmt(table=table, columns=cols, rows=rows)

    # ─── CREATE INDEX ─────────────────────────────────────────────────
    def _create_index(self):
        uniq = self._eat_if(TT.UNIQUE)
        self._eat(TT.INDEX)
        idx_name = self._eat_ident()
        self._eat(TT.ON)
        tbl = self._eat_ident()
        self._eat(TT.LPAREN)
        cols = self._ident_list()
        self._eat(TT.RPAREN)
        return CreateIndexStmt(name=idx_name, table=tbl, columns=cols, unique=uniq)

    # ─── ALTER TABLE ──────────────────────────────────────────────────
    def _alter(self):
        self._eat(TT.ALTER); self._eat(TT.TABLE)
        tbl = self._eat_ident()
        if self._peek() == TT.ADD:
            self._eat(TT.ADD)
            self._eat_if(TT.COLUMN)   # COLUMN keyword is optional
            return AlterAddCol(table=tbl, col=self._column_def())
        if self._peek() == TT.DROP:
            self._eat(TT.DROP)
            self._eat_if(TT.COLUMN)
            return AlterDropCol(table=tbl, col=self._eat_ident())
        if self._peek() == TT.RENAME:
            self._eat(TT.RENAME)
            if self._peek() == TT.TO:
                # rename table
                self._eat(TT.TO)
                return AlterRenameTable(old=tbl, new=self._eat_ident())
            # rename column
            old = self._eat_ident()
            self._eat(TT.TO)
            new = self._eat_ident()
            return AlterRenameCol(table=tbl, old=old, new=new)
        raise ParseError("expected ADD / DROP / RENAME", self._cur())

    # ─── UPDATE ───────────────────────────────────────────────────────
    def _update(self):
        self._eat(TT.UPDATE)
        tbl = self._eat_ident()
        self._eat(TT.SET)
        sets = []
        while True:
            col = self._eat_ident()
            self._eat(TT.EQ)
            val = self._expr()
            sets.append((col, val))
            if not self._eat_if(TT.COMMA):
                break
        w = self._where() if self._peek() == TT.WHERE else None
        return UpdateStmt(table=tbl, sets=sets, where_=w)

    # ─── DELETE ───────────────────────────────────────────────────────
    def _delete(self):
        self._eat(TT.DELETE); self._eat(TT.FROM)
        tbl = self._eat_ident()
        w = self._where() if self._peek() == TT.WHERE else None
        return DeleteStmt(table=tbl, where_=w)

    # ─── SELECT ───────────────────────────────────────────────────────
    def _select(self) -> SelectStmt:
        self._eat(TT.SELECT)
        distinct = self._eat_if(TT.DISTINCT)

        # column list
        cols: list[SelectCol | StarExpr] = []
        while True:
            if self._peek() == TT.STAR:
                self._advance()
                cols.append(StarExpr())
            else:
                e = self._expr()
                alias = None
                if self._eat_if(TT.AS):
                    alias = self._eat_ident()
                elif self._peek() == TT.IDENT:
                    # implicit alias (no AS)
                    alias = self._eat_ident()
                cols.append(SelectCol(expr=e, alias=alias))
            if not self._eat_if(TT.COMMA):
                break

        # FROM
        from_ = None
        if self._eat_if(TT.FROM):
            from_ = self._from_clause()

        # WHERE
        where_ = self._where() if self._peek() == TT.WHERE else None

        # GROUP BY
        group_by = []
        if self._peek() == TT.GROUP:
            self._eat(TT.GROUP); self._eat(TT.BY)
            group_by.append(self._expr())
            while self._eat_if(TT.COMMA):
                group_by.append(self._expr())

        # HAVING
        having = None
        if self._eat_if(TT.HAVING):
            having = self._expr()

        # ORDER BY
        order_by = []
        if self._peek() == TT.ORDER:
            self._eat(TT.ORDER); self._eat(TT.BY)
            order_by.append(self._order_item())
            while self._eat_if(TT.COMMA):
                order_by.append(self._order_item())

        # LIMIT / OFFSET
        limit = offset = None
        if self._eat_if(TT.LIMIT):
            limit = self._expr()
        if self._eat_if(TT.OFFSET):
            offset = self._expr()

        return SelectStmt(columns=cols, from_=from_, where_=where_,
                          group_by=group_by, having=having, order_by=order_by,
                          limit=limit, offset=offset, distinct=distinct)

    def _from_clause(self):
        tbl = self._from_table()
        # chained JOINs
        while self._peek() in (TT.JOIN, TT.INNER, TT.LEFT, TT.RIGHT):
            jt = "INNER"
            if self._peek() in (TT.INNER, TT.LEFT, TT.RIGHT):
                jt = self._advance().type.name
            self._eat(TT.JOIN)
            right = self._from_table()
            self._eat(TT.ON)
            on = self._expr()
            tbl = JoinClause(left=tbl, join_type=jt, right=right, on=on)
        return tbl

    def _from_table(self) -> FromTable:
        name = self._eat_ident()
        alias = None
        if self._eat_if(TT.AS):
            alias = self._eat_ident()
        elif self._peek() == TT.IDENT:
            alias = self._eat_ident()
        return FromTable(name=name, alias=alias)

    def _where(self) -> Expr:
        self._eat(TT.WHERE)
        return self._expr()

    def _order_item(self) -> OrderItem:
        e = self._expr()
        asc = True
        if self._eat_if(TT.ASC):   asc = True
        elif self._eat_if(TT.DESC): asc = False
        return OrderItem(expr=e, asc=asc)

    # ─── EXPRESSIONS (precedence climbing) ───────────────────────────
    # Precedence (low → high):
    #   OR
    #   AND
    #   NOT
    #   comparison:  = != < > <= > LIKE IN BETWEEN IS
    #   addition:    + -
    #   multiply:    * / %
    #   unary:       - NOT
    #   atom:        literal, column, func, (, CASE, subselect

    def _expr(self) -> Expr:
        return self._or()

    def _or(self) -> Expr:
        left = self._and()
        while self._eat_if(TT.OR):
            left = BinOp("OR", left, self._and())
        return left

    def _and(self) -> Expr:
        left = self._not()
        while self._eat_if(TT.AND):
            left = BinOp("AND", left, self._not())
        return left

    def _not(self) -> Expr:
        if self._eat_if(TT.NOT):
            return UnaryNot(self._not())
        return self._comparison()

    def _comparison(self) -> Expr:
        left = self._addition()
        # IS [NOT] NULL
        if self._peek() == TT.IS:
            self._eat(TT.IS)
            neg = self._eat_if(TT.NOT)
            self._eat(TT.NULL)
            return IsNullExpr(expr=left, negated=neg)
        # [NOT] LIKE / IN / BETWEEN
        neg = False
        if self._peek() == TT.NOT:
            # lookahead
            save = self.pos
            self._advance()
            if self._peek() in (TT.LIKE, TT.IN, TT.BETWEEN):
                neg = True
            else:
                self.pos = save
        if self._peek() == TT.LIKE:
            self._eat(TT.LIKE)
            return LikeExpr(expr=left, pattern=self._addition(), negated=neg)
        if self._peek() == TT.IN:
            self._eat(TT.IN); self._eat(TT.LPAREN)
            if self._peek() == TT.SELECT:
                sub = self._select()
                self._eat(TT.RPAREN)
                return InExpr(expr=left, values=None, subsel=sub, negated=neg)
            vals = [self._expr()]
            while self._eat_if(TT.COMMA):
                vals.append(self._expr())
            self._eat(TT.RPAREN)
            return InExpr(expr=left, values=vals, subsel=None, negated=neg)
        if self._peek() == TT.BETWEEN:
            self._eat(TT.BETWEEN)
            lo = self._addition()
            self._eat(TT.AND)
            hi = self._addition()
            return BetweenExpr(expr=left, low=lo, high=hi, negated=neg)
        # binary comparisons
        CMP = {TT.EQ: "=", TT.NEQ: "!=", TT.LT: "<", TT.GT: ">", TT.LTE: "<=", TT.GTE: ">="}
        if self._peek() in CMP:
            op = CMP[self._advance().type]
            return BinOp(op, left, self._addition())
        return left

    def _addition(self) -> Expr:
        left = self._multiply()
        while self._peek() in (TT.PLUS, TT.MINUS):
            op = '+' if self._advance().type == TT.PLUS else '-'
            left = BinOp(op, left, self._multiply())
        return left

    def _multiply(self) -> Expr:
        left = self._unary()
        while self._peek() in (TT.STAR, TT.SLASH, TT.PERCENT):
            t = self._advance().type
            op = {'STAR': '*', 'SLASH': '/', 'PERCENT': '%'}[t.name]
            left = BinOp(op, left, self._unary())
        return left

    def _unary(self) -> Expr:
        if self._eat_if(TT.MINUS):
            return UnaryMinus(self._unary())
        if self._eat_if(TT.NOT):
            return UnaryNot(self._unary())
        return self._atom()

    def _atom(self) -> Expr:
        t = self._cur()
        # literals
        if t.type == TT.INT:    self._advance(); return LitInt(t.value)
        if t.type == TT.FLOAT:  self._advance(); return LitFloat(t.value)
        if t.type == TT.STR:    self._advance(); return LitStr(t.value)
        if t.type == TT.TRUE:   self._advance(); return LitBool(True)
        if t.type == TT.FALSE:  self._advance(); return LitBool(False)
        if t.type == TT.NULL:   self._advance(); return LitNull()

        # CASE
        if t.type == TT.CASE:
            return self._case_expr()

        # EXISTS
        if t.type == TT.EXISTS:
            self._advance(); self._eat(TT.LPAREN)
            sub = self._select()
            self._eat(TT.RPAREN)
            return ExistsExpr(sub)

        # sub-select  ( SELECT … )
        if t.type == TT.LPAREN:
            self._advance()
            if self._peek() == TT.SELECT:
                sub = self._select()
                self._eat(TT.RPAREN)
                # cast?
                return self._maybe_cast(SubSelectExpr(sub))
            # grouped expression
            e = self._expr()
            self._eat(TT.RPAREN)
            return self._maybe_cast(e)

        # identifier — could be column ref or function call
        if t.type == TT.IDENT or t.type in KEYWORD_AS_IDENT:
            name = self._advance().value
            # function call?
            if self._peek() == TT.LPAREN:
                self._advance()
                distinct = self._eat_if(TT.DISTINCT)
                if self._peek() == TT.RPAREN:
                    self._advance()
                    return self._maybe_cast(FuncCall(name=name, args=[], distinct=distinct))
                if self._peek() == TT.STAR:
                    self._advance()
                    self._eat(TT.RPAREN)
                    return self._maybe_cast(FuncCall(name=name, args=[StarExpr()], distinct=distinct))
                args = [self._expr()]
                while self._eat_if(TT.COMMA):
                    args.append(self._expr())
                self._eat(TT.RPAREN)
                return self._maybe_cast(FuncCall(name=name, args=args, distinct=distinct))
            # qualified column?  tbl.col
            if self._eat_if(TT.DOT):
                col = self._eat_ident()
                return self._maybe_cast(ColRef(table=name, name=col))
            return self._maybe_cast(ColRef(table=None, name=name))

        # aggregate keywords used as function names: COUNT(…), SUM(…), etc.
        AGG_KW = {TT.COUNT: "COUNT", TT.SUM: "SUM", TT.AVG: "AVG",
                  TT.MIN: "MIN", TT.MAX: "MAX"}
        if t.type in AGG_KW:
            fname = AGG_KW[t.type]
            self._advance()
            self._eat(TT.LPAREN)
            distinct = self._eat_if(TT.DISTINCT)
            if self._peek() == TT.STAR:
                self._advance(); self._eat(TT.RPAREN)
                return self._maybe_cast(FuncCall(name=fname, args=[StarExpr()], distinct=distinct))
            args = [self._expr()]
            while self._eat_if(TT.COMMA):
                args.append(self._expr())
            self._eat(TT.RPAREN)
            return self._maybe_cast(FuncCall(name=fname, args=args, distinct=distinct))

        raise ParseError("expected expression", t)

    def _maybe_cast(self, e: Expr) -> Expr:
        if self._eat_if(TT.DCOLON):
            TYPE_KW = {TT.T_INT: "INT", TT.T_BIGINT: "BIGINT", TT.T_REAL: "REAL",
                       TT.T_TEXT: "TEXT", TT.T_BLOB: "BLOB", TT.T_BOOL: "BOOL"}
            if self._peek() in TYPE_KW:
                to = TYPE_KW[self._advance().type]
                return CastExpr(expr=e, to=to)
            raise ParseError("expected type after ::", self._cur())
        return e

    def _case_expr(self) -> CaseExpr:
        self._eat(TT.CASE)
        operand = None
        # simple CASE?  — CASE <expr> WHEN …
        if self._peek() != TT.WHEN:
            operand = self._expr()
        whens = []
        while self._eat_if(TT.WHEN):
            cond = self._expr()
            self._eat(TT.THEN)
            res  = self._expr()
            whens.append((cond, res))
        else_ = None
        if self._eat_if(TT.ELSE):
            else_ = self._expr()
        self._eat(TT.END)
        return CaseExpr(operand=operand, whens=whens, else_=else_)


# Keywords that may also appear as identifiers (column names, etc.)
KEYWORD_AS_IDENT = {
    TT.KEY, TT.INDEX, TT.END, TT.LEFT, TT.RIGHT, TT.ON,
    TT.SET, TT.ADD, TT.COLUMN, TT.TO, TT.PRIMARY, TT.UNIQUE,
    TT.COUNT, TT.SUM, TT.AVG, TT.MIN, TT.MAX,
    TT.T_INT, TT.T_BIGINT, TT.T_REAL, TT.T_TEXT, TT.T_BLOB, TT.T_BOOL,
}


# ════════════════════════════════════════════════════════════════════════════
# 4.  STORAGE ENGINE  (in-memory + B-Tree indexes)
# ════════════════════════════════════════════════════════════════════════════

class StorageError(Exception): pass


class BTreeIndex:
    """Minimal B-Tree index.  Maps index-key tuples → list of row-ids."""
    ORDER = 64   # max keys per node

    def __init__(self, table: str, columns: list[str], unique: bool):
        self.table   = table
        self.columns = columns
        self.unique  = unique
        # Simple sorted-list implementation (production would use real B-tree nodes)
        self._data: list[tuple[tuple, list[int]]] = []   # sorted by key

    def _find(self, key: tuple) -> int:
        """Binary search → insertion point."""
        lo, hi = 0, len(self._data)
        while lo < hi:
            mid = (lo + hi) // 2
            if self._data[mid][0] < key:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def insert(self, key: tuple, row_id: int):
        pos = self._find(key)
        if pos < len(self._data) and self._data[pos][0] == key:
            if self.unique:
                raise StorageError(
                    f"UNIQUE constraint violated on index '{self.table}({', '.join(self.columns)})'")
            self._data[pos][1].append(row_id)
        else:
            self._data.insert(pos, (key, [row_id]))

    def delete(self, key: tuple, row_id: int):
        pos = self._find(key)
        if pos < len(self._data) and self._data[pos][0] == key:
            ids = self._data[pos][1]
            if row_id in ids:
                ids.remove(row_id)
            if not ids:
                del self._data[pos]

    def lookup(self, key: tuple) -> list[int]:
        pos = self._find(key)
        if pos < len(self._data) and self._data[pos][0] == key:
            return list(self._data[pos][1])
        return []

    def all_row_ids(self) -> list[int]:
        """Full scan — returns all row ids in key order."""
        out = []
        for _, ids in self._data:
            out.extend(ids)
        return out

    def clear(self):
        self._data.clear()


class Table:
    """One table: schema + row store + indexes."""
    def __init__(self, name: str, columns: list[ColumnDef],
                 primary_key: list[str] | None = None):
        self.name        = name
        self.columns     = columns
        self.col_names   = [c.name for c in columns]
        self.primary_key = primary_key or []
        self.rows: dict[int, list[Any]] = {}   # row_id → values
        self._next_id    = 0
        self.indexes: dict[str, BTreeIndex] = {}

        # Auto-create PK index
        if self.primary_key:
            idx = BTreeIndex(name, self.primary_key, unique=True)
            self.indexes[f"__pk_{name}"] = idx

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def col_index(self, name: str) -> int:
        try:
            return self.col_names.index(name)
        except ValueError:
            raise StorageError(f"column '{name}' not found in table '{self.name}'")

    def insert_row(self, values: list[Any]) -> int:
        rid = self._new_id()
        # NOT NULL / UNIQUE checks done by executor
        self.rows[rid] = values
        # update indexes
        for idx in self.indexes.values():
            key = tuple(values[self.col_index(c)] for c in idx.columns)
            idx.insert(key, rid)
        return rid

    def delete_row(self, rid: int):
        if rid not in self.rows:
            return
        values = self.rows[rid]
        for idx in self.indexes.values():
            key = tuple(values[self.col_index(c)] for c in idx.columns)
            idx.delete(key, rid)
        del self.rows[rid]

    def update_row(self, rid: int, new_values: list[Any]):
        old = self.rows.get(rid)
        if old is None:
            return
        # remove old index entries
        for idx in self.indexes.values():
            key = tuple(old[self.col_index(c)] for c in idx.columns)
            idx.delete(key, rid)
        self.rows[rid] = new_values
        # add new
        for idx in self.indexes.values():
            key = tuple(new_values[self.col_index(c)] for c in idx.columns)
            idx.insert(key, rid)


class Catalog:
    """The system catalog: all tables and indexes."""
    def __init__(self):
        self.tables: dict[str, Table] = {}

    def get_table(self, name: str) -> Table:
        key = name.lower()
        if key not in self.tables:
            raise StorageError(f"table '{name}' does not exist")
        return self.tables[key]

    def has_table(self, name: str) -> bool:
        return name.lower() in self.tables


# ════════════════════════════════════════════════════════════════════════════
# 5.  TRANSACTION SUPPORT  (snapshot isolation via copy-on-write)
# ════════════════════════════════════════════════════════════════════════════

class TransactionManager:
    """Lightweight snapshot-based transactions.
    BEGIN  → save a deep copy of the catalog.
    COMMIT → discard the snapshot (changes are live).
    ROLLBACK → restore from snapshot.
    """
    def __init__(self):
        self.active   = False
        self._snapshot: Catalog | None = None

    def begin(self, catalog: Catalog):
        if self.active:
            raise StorageError("nested transactions not supported; commit or rollback first")
        self._snapshot = copy.deepcopy(catalog)
        self.active = True

    def commit(self):
        if not self.active:
            raise StorageError("no active transaction")
        self._snapshot = None
        self.active = False

    def rollback(self, catalog: Catalog) -> Catalog:
        if not self.active:
            raise StorageError("no active transaction")
        restored = self._snapshot
        self._snapshot = None
        self.active = False
        # Copy state back
        catalog.tables = restored.tables
        return catalog


# ════════════════════════════════════════════════════════════════════════════
# 6.  EXECUTOR  (walks the AST, runs queries against the catalog)
# ════════════════════════════════════════════════════════════════════════════

class ExecError(Exception): pass


# A Row in flight is just a dict: { "table.col" | "col" → value }
Row = dict[str, Any]


def _coerce(value: Any, col_type: str) -> Any:
    """Best-effort type coercion on INSERT."""
    if value is None:
        return None
    try:
        if col_type in ("INT", "BIGINT"):  return int(value)
        if col_type == "REAL":             return float(value)
        if col_type == "TEXT":             return str(value)
        if col_type == "BOOL":
            if isinstance(value, bool):    return value
            if isinstance(value, int):     return bool(value)
            if isinstance(value, str):     return value.lower() in ("true", "1")
        if col_type == "BLOB":             return value
    except (ValueError, TypeError):
        pass
    return value


class Executor:
    def __init__(self):
        self.catalog = Catalog()
        self.txn     = TransactionManager()

    # ── dispatch ──────────────────────────────────────────────────────
    def execute(self, stmt) -> tuple[list[str], list[list[Any]]]:
        """Returns (column_headers, result_rows).  DDL returns empty rows."""
        match stmt:
            case CreateTableStmt():    return self._create_table(stmt)
            case DropTableStmt():      return self._drop_table(stmt)
            case InsertStmt():         return self._insert(stmt)
            case SelectStmt():         return self._select(stmt)
            case UpdateStmt():         return self._update(stmt)
            case DeleteStmt():         return self._delete(stmt)
            case CreateIndexStmt():    return self._create_index(stmt)
            case DropIndexStmt():      return self._drop_index(stmt)
            case AlterAddCol():        return self._alter_add_col(stmt)
            case AlterDropCol():       return self._alter_drop_col(stmt)
            case AlterRenameCol():     return self._alter_rename_col(stmt)
            case AlterRenameTable():   return self._alter_rename_table(stmt)
            case BeginStmt():
                self.txn.begin(self.catalog)
                return [], []
            case CommitStmt():
                self.txn.commit()
                return [], []
            case RollbackStmt():
                self.txn.rollback(self.catalog)
                return [], []
            case _:
                raise ExecError(f"unknown statement type: {type(stmt).__name__}")

    # ─── CREATE TABLE ───────────────────────────────────────────────
    def _create_table(self, s: CreateTableStmt):
        key = s.name.lower()
        if self.catalog.has_table(key):
            if s.if_not_exists:
                return [], []
            raise ExecError(f"table '{s.name}' already exists")
        tbl = Table(s.name, s.columns, s.primary_key)
        self.catalog.tables[key] = tbl
        return [], []

    # ─── DROP TABLE ─────────────────────────────────────────────────
    def _drop_table(self, s: DropTableStmt):
        key = s.name.lower()
        if key not in self.catalog.tables:
            if s.if_exists:
                return [], []
            raise ExecError(f"table '{s.name}' does not exist")
        del self.catalog.tables[key]
        return [], []

    # ─── INSERT ─────────────────────────────────────────────────────
    def _insert(self, s: InsertStmt):
        tbl = self.catalog.get_table(s.table)
        # Determine column order
        if s.columns:
            col_order = [tbl.col_index(c) for c in s.columns]
        else:
            col_order = list(range(len(tbl.columns)))

        if len(col_order) == 0:
            raise ExecError("INSERT with no columns")

        count = 0
        for row_exprs in s.rows:
            if len(row_exprs) != len(col_order):
                raise ExecError(
                    f"INSERT: expected {len(col_order)} values, got {len(row_exprs)}")
            # Build full row with NULLs
            values: list[Any] = [None] * len(tbl.columns)
            for i, ci in enumerate(col_order):
                values[ci] = self._eval_expr(row_exprs[i], {})

            # Coerce types
            for i, col in enumerate(tbl.columns):
                values[i] = _coerce(values[i], col.col_type)

            # NOT NULL checks
            for i, col in enumerate(tbl.columns):
                if col.not_null and values[i] is None:
                    raise ExecError(
                        f"NOT NULL constraint failed: {tbl.name}.{col.name}")

            tbl.insert_row(values)
            count += 1
        return [], []  # INSERT returns no rows

    # ─── SELECT ─────────────────────────────────────────────────────
    def _select(self, s: SelectStmt) -> tuple[list[str], list[list[Any]]]:
        # 1) FROM → produce candidate rows
        rows: list[Row] = self._eval_from(s.from_)

        # 2) WHERE
        if s.where_:
            rows = [r for r in rows if self._truthy(self._eval_expr(s.where_, r))]

        # 3) GROUP BY / aggregates
        if s.group_by or self._has_aggregate(s.columns):
            # If no GROUP BY and no rows, produce one empty group so COUNT(*) → 0
            if not s.group_by and not rows:
                rows = [{"__group__": []}]
            else:
                rows = self._group_and_aggregate(s, rows)
            # HAVING
            if s.having:
                rows = [r for r in rows if self._truthy(self._eval_expr(s.having, r))]
            headers, result = self._project(s, rows)
        else:
            # no grouping — just project
            # HAVING without GROUP BY is an error in strict SQL but we allow it as a WHERE alias
            if s.having:
                rows = [r for r in rows if self._truthy(self._eval_expr(s.having, r))]
            headers, result = self._project(s, rows)

        # 4) DISTINCT
        if s.distinct:
            seen = set()
            unique = []
            for r in result:
                key = tuple(None if v is None else v for v in r)
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            result = unique

        # 5) ORDER BY
        if s.order_by:
            def sort_key(row_vals):
                keys = []
                row_dict = dict(zip(headers, row_vals))
                for item in s.order_by:
                    v = self._eval_expr(item.expr, row_dict)
                    # None sorts last
                    keys.append((v is None, v if v is not None else 0, item.asc))
                return keys

            def cmp_rows(a, b):
                ka, kb = sort_key(a), sort_key(b)
                for (a_none, av, asc), (b_none, bv, _) in zip(ka, kb):
                    if a_none != b_none:
                        return 1 if a_none else -1
                    if av == bv:
                        continue
                    try:
                        lt = av < bv
                    except TypeError:
                        lt = str(av) < str(bv)
                    if lt:
                        return -1 if asc else 1
                    else:
                        return 1 if asc else -1
                return 0

            import functools
            result.sort(key=functools.cmp_to_key(cmp_rows))

        # 6) OFFSET / LIMIT
        if s.offset:
            off = int(self._eval_expr(s.offset, {}))
            result = result[off:]
        if s.limit:
            lim = int(self._eval_expr(s.limit, {}))
            result = result[:lim]

        return headers, result

    # ─── UPDATE ─────────────────────────────────────────────────────
    def _update(self, s: UpdateStmt):
        tbl = self.catalog.get_table(s.table)
        count = 0
        for rid, vals in list(tbl.rows.items()):
            row = self._row_from_table(tbl, vals, None)
            if s.where_ and not self._truthy(self._eval_expr(s.where_, row)):
                continue
            new_vals = list(vals)
            for col_name, expr in s.sets:
                ci = tbl.col_index(col_name)
                new_vals[ci] = _coerce(self._eval_expr(expr, row), tbl.columns[ci].col_type)
            # NOT NULL check
            for i, col in enumerate(tbl.columns):
                if col.not_null and new_vals[i] is None:
                    raise ExecError(f"NOT NULL constraint failed: {tbl.name}.{col.name}")
            tbl.update_row(rid, new_vals)
            count += 1
        return [], []

    # ─── DELETE ─────────────────────────────────────────────────────
    def _delete(self, s: DeleteStmt):
        tbl = self.catalog.get_table(s.table)
        to_del = []
        for rid, vals in tbl.rows.items():
            row = self._row_from_table(tbl, vals, None)
            if s.where_ and not self._truthy(self._eval_expr(s.where_, row)):
                continue
            to_del.append(rid)
        for rid in to_del:
            tbl.delete_row(rid)
        return [], []

    # ─── CREATE INDEX ───────────────────────────────────────────────
    def _create_index(self, s: CreateIndexStmt):
        tbl = self.catalog.get_table(s.table)
        if s.name in tbl.indexes:
            raise ExecError(f"index '{s.name}' already exists")
        idx = BTreeIndex(s.table, s.columns, s.unique)
        # populate from existing rows
        for rid, vals in tbl.rows.items():
            key = tuple(vals[tbl.col_index(c)] for c in s.columns)
            idx.insert(key, rid)
        tbl.indexes[s.name] = idx
        return [], []

    # ─── DROP INDEX ─────────────────────────────────────────────────
    def _drop_index(self, s: DropIndexStmt):
        # scan all tables for the index
        for tbl in self.catalog.tables.values():
            if s.name in tbl.indexes:
                del tbl.indexes[s.name]
                return [], []
        if s.if_exists:
            return [], []
        raise ExecError(f"index '{s.name}' does not exist")

    # ─── ALTER ──────────────────────────────────────────────────────
    def _alter_add_col(self, s: AlterAddCol):
        tbl = self.catalog.get_table(s.table)
        if s.col.name in tbl.col_names:
            raise ExecError(f"column '{s.col.name}' already exists in '{tbl.name}'")
        tbl.columns.append(s.col)
        tbl.col_names.append(s.col.name)
        # pad existing rows
        for rid in tbl.rows:
            tbl.rows[rid].append(None)
        return [], []

    def _alter_drop_col(self, s: AlterDropCol):
        tbl = self.catalog.get_table(s.table)
        ci = tbl.col_index(s.col)
        # check not in PK
        if s.col in tbl.primary_key:
            raise ExecError(f"cannot drop primary key column '{s.col}'")
        # remove from indexes
        for iname in list(tbl.indexes):
            if s.col in tbl.indexes[iname].columns:
                del tbl.indexes[iname]
        # remove column
        tbl.columns.pop(ci)
        tbl.col_names.pop(ci)
        for rid in tbl.rows:
            tbl.rows[rid].pop(ci)
        return [], []

    def _alter_rename_col(self, s: AlterRenameCol):
        tbl = self.catalog.get_table(s.table)
        ci = tbl.col_index(s.old)
        tbl.columns[ci].name = s.new
        tbl.col_names[ci] = s.new
        return [], []

    def _alter_rename_table(self, s: AlterRenameTable):
        old_key = s.old.lower()
        new_key = s.new.lower()
        if old_key not in self.catalog.tables:
            raise ExecError(f"table '{s.old}' does not exist")
        if new_key in self.catalog.tables:
            raise ExecError(f"table '{s.new}' already exists")
        tbl = self.catalog.tables.pop(old_key)
        tbl.name = s.new
        self.catalog.tables[new_key] = tbl
        return [], []

    # ─── FROM clause evaluation ─────────────────────────────────────
    def _eval_from(self, from_: FromTable | JoinClause | None) -> list[Row]:
        if from_ is None:
            return [{}]   # SELECT without FROM — one empty row

        if isinstance(from_, FromTable):
            return self._scan_table(from_)

        if isinstance(from_, JoinClause):
            left_rows  = self._eval_from(from_.left)
            right_rows = self._eval_from(from_.right)
            return self._do_join(from_, left_rows, right_rows)

        raise ExecError(f"unknown FROM type: {type(from_)}")

    def _scan_table(self, ft: FromTable) -> list[Row]:
        tbl = self.catalog.get_table(ft.name)
        prefix = ft.alias or tbl.name
        out = []
        for vals in tbl.rows.values():
            row: Row = {}
            for i, col in enumerate(tbl.columns):
                row[f"{prefix}.{col.name}"] = vals[i]
                row[col.name] = vals[i]   # also unqualified
            out.append(row)
        return out

    def _row_from_table(self, tbl: Table, vals: list, alias: str | None) -> Row:
        prefix = alias or tbl.name
        row: Row = {}
        for i, col in enumerate(tbl.columns):
            row[f"{prefix}.{col.name}"] = vals[i]
            row[col.name] = vals[i]
        return row

    # ─── JOIN ───────────────────────────────────────────────────────
    def _do_join(self, jc: JoinClause,
                 left_rows: list[Row], right_rows: list[Row]) -> list[Row]:
        result = []
        jt = jc.join_type

        for lr in left_rows:
            matched = False
            for rr in right_rows:
                merged = {**lr, **rr}
                if self._truthy(self._eval_expr(jc.on, merged)):
                    result.append(merged)
                    matched = True
            if jt == "LEFT" and not matched:
                # fill right side with NULLs
                null_right: Row = {}
                # figure out right-side keys
                if isinstance(jc.right, FromTable):
                    rt = self.catalog.get_table(jc.right.name)
                    prefix = jc.right.alias or rt.name
                    for col in rt.columns:
                        null_right[f"{prefix}.{col.name}"] = None
                        null_right[col.name] = None
                result.append({**lr, **null_right})

        if jt == "RIGHT":
            # also need unmatched right rows with left=NULL
            for rr in right_rows:
                matched = any(
                    self._truthy(self._eval_expr(jc.on, {**lr, **rr}))
                    for lr in left_rows
                )
                if not matched:
                    null_left: Row = {}
                    if isinstance(jc.left, FromTable):
                        lt = self.catalog.get_table(jc.left.name)
                        prefix = jc.left.alias or lt.name
                        for col in lt.columns:
                            null_left[f"{prefix}.{col.name}"] = None
                            null_left[col.name] = None
                    result.append({**null_left, **rr})

        return result

    # ─── GROUP BY / Aggregates ──────────────────────────────────────
    def _has_aggregate(self, cols) -> bool:
        AGG = {"COUNT", "SUM", "AVG", "MIN", "MAX"}
        def check(e):
            if isinstance(e, FuncCall) and e.name.upper() in AGG:
                return True
            if isinstance(e, SelectCol):
                return check(e.expr)
            if isinstance(e, BinOp):
                return check(e.left) or check(e.right)
            if isinstance(e, UnaryNot) or isinstance(e, UnaryMinus):
                return check(e.expr)
            if isinstance(e, CaseExpr):
                return any(check(c) or check(r) for c, r in e.whens) or (e.else_ and check(e.else_))
            return False
        return any(check(c) for c in cols)

    def _group_and_aggregate(self, s: SelectStmt, rows: list[Row]) -> list[Row]:
        """Group rows, then for each group produce a single output Row
           that contains both the group-key columns and placeholders
           __agg_<i> that the expression evaluator will fill."""
        from collections import OrderedDict
        groups: OrderedDict[tuple, list[Row]] = OrderedDict()

        for r in rows:
            if s.group_by:
                key = tuple(self._eval_expr(e, r) for e in s.group_by)
            else:
                key = ()
            groups.setdefault(key, []).append(r)

        out = []
        for key, group_rows in groups.items():
            # The output row carries the group keys + a special __group__ list
            # so that aggregate functions can iterate over it.
            representative = dict(group_rows[0])
            representative["__group__"] = group_rows
            if s.group_by:
                for i, e in enumerate(s.group_by):
                    representative[f"__gk_{i}"] = key[i]
            out.append(representative)
        return out

    # ─── Projection ─────────────────────────────────────────────────
    def _project(self, s: SelectStmt, rows: list[Row]) -> tuple[list[str], list[list[Any]]]:
        headers: list[str] = []
        result: list[list[Any]] = []

        # Expand * first to get headers
        for col in s.columns:
            if isinstance(col, StarExpr):
                # grab all column names from FROM
                if s.from_:
                    for name in self._star_columns(s.from_):
                        if name not in headers:
                            headers.append(name)
                else:
                    headers.append("*")
            elif isinstance(col, SelectCol):
                h = col.alias or self._expr_header(col.expr)
                headers.append(h)

        for row in rows:
            vals = []
            for col in s.columns:
                if isinstance(col, StarExpr):
                    for name in self._star_columns(s.from_) if s.from_ else ["*"]:
                        vals.append(row.get(name))
                elif isinstance(col, SelectCol):
                    vals.append(self._eval_expr(col.expr, row))
            result.append(vals)

        return headers, result

    def _star_columns(self, from_) -> list[str]:
        """Get column names for * expansion."""
        if isinstance(from_, FromTable):
            tbl = self.catalog.get_table(from_.name)
            return [c.name for c in tbl.columns]
        if isinstance(from_, JoinClause):
            return self._star_columns(from_.left) + self._star_columns(from_.right)
        return []

    def _expr_header(self, e: Expr) -> str:
        if isinstance(e, ColRef):   return e.name
        if isinstance(e, FuncCall): return f"{e.name}({', '.join(self._expr_header(a) for a in e.args)})"
        if isinstance(e, LitInt):   return str(e.val)
        if isinstance(e, LitStr):   return f"'{e.val}'"
        return "?"

    # ─── Expression Evaluator ───────────────────────────────────────
    def _eval_expr(self, e: Expr, row: Row) -> Any:
        match e:
            case LitInt(v):     return v
            case LitFloat(v):   return v
            case LitStr(v):     return v
            case LitBool(v):    return v
            case LitNull():     return None
            case StarExpr():    return None

            case ColRef(table=tbl, name=name):
                if tbl:
                    k = f"{tbl}.{name}"
                    if k in row:
                        return row[k]
                # unqualified
                if name in row:
                    return row[name]
                # case-insensitive fallback
                low = name.lower()
                for rk, rv in row.items():
                    if rk.lower() == low or rk.lower().endswith(f".{low}"):
                        return rv
                return None   # missing column → NULL (lenient)

            case BinOp(op=op, left=left, right=right):
                return self._eval_binop(op, left, right, row)

            case UnaryNot(expr=inner):
                v = self._eval_expr(inner, row)
                if v is None:  return None
                return not self._truthy(v)

            case UnaryMinus(expr=inner):
                v = self._eval_expr(inner, row)
                if v is None:  return None
                return -v

            case FuncCall(name=name, args=args, distinct=distinct):
                return self._eval_func(name, args, distinct, row)

            case CaseExpr(operand=operand, whens=whens, else_=else_):
                if operand is not None:
                    op_val = self._eval_expr(operand, row)
                    for cond, res in whens:
                        if self._eval_expr(cond, row) == op_val:
                            return self._eval_expr(res, row)
                else:
                    for cond, res in whens:
                        if self._truthy(self._eval_expr(cond, row)):
                            return self._eval_expr(res, row)
                return self._eval_expr(else_, row) if else_ else None

            case IsNullExpr(expr=inner, negated=neg):
                v = self._eval_expr(inner, row)
                is_null = (v is None)
                return (not is_null) if neg else is_null

            case LikeExpr(expr=inner, pattern=pat, negated=neg):
                v = self._eval_expr(inner, row)
                p = self._eval_expr(pat, row)
                if v is None or p is None:
                    return None
                match_result = self._like_match(str(v), str(p))
                return (not match_result) if neg else match_result

            case InExpr(expr=inner, values=vals, subsel=sub, negated=neg):
                v = self._eval_expr(inner, row)
                if v is None:
                    return None
                if vals is not None:
                    items = [self._eval_expr(x, row) for x in vals]
                elif sub is not None:
                    _, sub_rows = self._select(sub)
                    items = [r[0] for r in sub_rows] if sub_rows else []
                else:
                    items = []
                result = v in items
                return (not result) if neg else result

            case BetweenExpr(expr=inner, low=lo, high=hi, negated=neg):
                v  = self._eval_expr(inner, row)
                lv = self._eval_expr(lo, row)
                hv = self._eval_expr(hi, row)
                if v is None or lv is None or hv is None:
                    return None
                result = lv <= v <= hv
                return (not result) if neg else result

            case ExistsExpr(subsel=sub):
                _, sub_rows = self._select(sub)
                return len(sub_rows) > 0

            case SubSelectExpr(subsel=sub):
                _, sub_rows = self._select(sub)
                if sub_rows:
                    return sub_rows[0][0] if sub_rows[0] else None
                return None

            case CastExpr(expr=inner, to=to):
                v = self._eval_expr(inner, row)
                return _coerce(v, to)

            case _:
                raise ExecError(f"cannot evaluate expression: {type(e).__name__}")

    def _eval_binop(self, op: str, left: Expr, right: Expr, row: Row) -> Any:
        # Short-circuit for AND / OR
        if op == "AND":
            lv = self._eval_expr(left, row)
            if not self._truthy(lv):
                return False
            return self._truthy(self._eval_expr(right, row))
        if op == "OR":
            lv = self._eval_expr(left, row)
            if self._truthy(lv):
                return True
            return self._truthy(self._eval_expr(right, row))

        lv = self._eval_expr(left, row)
        rv = self._eval_expr(right, row)

        # NULL propagation for arithmetic
        if op in ("+", "-", "*", "/", "%"):
            if lv is None or rv is None:
                return None
            if op == "+":
                if isinstance(lv, str) or isinstance(rv, str):
                    return str(lv) + str(rv)   # string concat with +
                return lv + rv
            if op == "-": return lv - rv
            if op == "*": return lv * rv
            if op == "/":
                if rv == 0:
                    raise ExecError("division by zero")
                if isinstance(lv, int) and isinstance(rv, int):
                    return lv // rv
                return lv / rv
            if op == "%":
                if rv == 0:
                    raise ExecError("modulo by zero")
                return lv % rv

        # Comparisons — NULL propagation
        if op in ("=", "!=", "<", ">", "<=", ">="):
            if lv is None or rv is None:
                if op == "=":  return lv is None and rv is None
                if op == "!=": return not (lv is None and rv is None)
                return None
            # type-flexible comparison
            try:
                if op == "=":  return lv == rv
                if op == "!=": return lv != rv
                if op == "<":  return lv < rv
                if op == ">":  return lv > rv
                if op == "<=": return lv <= rv
                if op == ">=": return lv >= rv
            except TypeError:
                return str(lv) == str(rv) if op == "=" else str(lv) != str(rv)

        raise ExecError(f"unknown operator: {op}")

    # ─── Aggregate functions ────────────────────────────────────────
    def _eval_func(self, name: str, args: list[Expr], distinct: bool, row: Row) -> Any:
        NAME = name.upper()
        AGG  = {"COUNT", "SUM", "AVG", "MIN", "MAX"}

        # Non-aggregate scalar functions
        if NAME == "ABS":
            v = self._eval_expr(args[0], row)
            return None if v is None else abs(v)
        if NAME == "LOWER":
            v = self._eval_expr(args[0], row)
            return None if v is None else str(v).lower()
        if NAME == "UPPER":
            v = self._eval_expr(args[0], row)
            return None if v is None else str(v).upper()
        if NAME == "LENGTH" or NAME == "LEN":
            v = self._eval_expr(args[0], row)
            return None if v is None else len(str(v))
        if NAME == "TRIM":
            v = self._eval_expr(args[0], row)
            return None if v is None else str(v).strip()
        if NAME == "COALESCE":
            for a in args:
                v = self._eval_expr(a, row)
                if v is not None:
                    return v
            return None
        if NAME == "IFNULL" or NAME == "NVL":
            v = self._eval_expr(args[0], row)
            return v if v is not None else self._eval_expr(args[1], row)
        if NAME == "NULLIF":
            a = self._eval_expr(args[0], row)
            b = self._eval_expr(args[1], row)
            return None if a == b else a
        if NAME == "TYPEOF":
            v = self._eval_expr(args[0], row)
            if v is None:          return "NULL"
            if isinstance(v, bool): return "BOOL"
            if isinstance(v, int):  return "INT"
            if isinstance(v, float):return "REAL"
            return "TEXT"
        if NAME == "ROUND":
            v = self._eval_expr(args[0], row)
            n = int(self._eval_expr(args[1], row)) if len(args) > 1 else 0
            return None if v is None else round(float(v), n)
        if NAME == "SUBSTR" or NAME == "SUBSTRING":
            s = self._eval_expr(args[0], row)
            start = int(self._eval_expr(args[1], row))
            length = int(self._eval_expr(args[2], row)) if len(args) > 2 else None
            if s is None: return None
            s = str(s)
            # SQL is 1-indexed
            idx = max(start - 1, 0)
            return s[idx:idx+length] if length else s[idx:]
        if NAME == "REPLACE":
            s = self._eval_expr(args[0], row)
            old = self._eval_expr(args[1], row)
            new = self._eval_expr(args[2], row)
            if s is None: return None
            return str(s).replace(str(old), str(new))
        if NAME == "CONCAT":
            parts = [self._eval_expr(a, row) for a in args]
            if any(p is None for p in parts): return None
            return "".join(str(p) for p in parts)

        # Aggregate functions — need __group__ in row
        if NAME not in AGG:
            raise ExecError(f"unknown function: {name}")

        group = row.get("__group__", [row])   # single-row "group" if no GROUP BY

        # Collect values
        if args and not isinstance(args[0], StarExpr):
            values = [self._eval_expr(args[0], r) for r in group]
        else:
            values = [1] * len(group)   # COUNT(*)

        if distinct:
            values = list(dict.fromkeys(values))   # preserve order, deduplicate

        if NAME == "COUNT":
            if args and not isinstance(args[0], StarExpr):
                return sum(1 for v in values if v is not None)
            return len(group)
        if NAME == "SUM":
            non_null = [v for v in values if v is not None]
            return sum(non_null) if non_null else None
        if NAME == "AVG":
            non_null = [v for v in values if v is not None]
            return (sum(non_null) / len(non_null)) if non_null else None
        if NAME == "MIN":
            non_null = [v for v in values if v is not None]
            return min(non_null) if non_null else None
        if NAME == "MAX":
            non_null = [v for v in values if v is not None]
            return max(non_null) if non_null else None

        raise ExecError(f"unknown aggregate: {NAME}")

    # ─── LIKE matching ──────────────────────────────────────────────
    @staticmethod
    def _like_match(value: str, pattern: str) -> bool:
        """SQL LIKE: % = any sequence, _ = any single char. Case-insensitive."""
        # Convert LIKE pattern to regex
        regex = "^"
        for ch in pattern:
            if ch == "%":  regex += ".*"
            elif ch == "_": regex += "."
            else:           regex += re.escape(ch)
        regex += "$"
        return bool(re.match(regex, value, re.IGNORECASE))

    # ─── Helpers ────────────────────────────────────────────────────
    @staticmethod
    def _truthy(v: Any) -> bool:
        if v is None:  return False
        if isinstance(v, bool): return v
        if isinstance(v, (int, float)): return v != 0
        if isinstance(v, str): return len(v) > 0
        return bool(v)


# ════════════════════════════════════════════════════════════════════════════
# 7.  REPL  (Read-Eval-Print Loop)
# ════════════════════════════════════════════════════════════════════════════

BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║                                                                       ║
║     Q Q Q Q Q     F F F F F                                           ║
║     Q     Q       F F F F F                                           ║
║     Q     Q       F F                                                 ║
║     Q     Q       F F                                                 ║
║     Q   Q Q       F F                                                 ║
║      Q Q Q Q                                                          ║
║                                                                       ║
║         Q U E R Y F O R G E   v0.1                                    ║
║         A Database Engine — Built From Scratch                        ║
║                                                                       ║
║  Commands:                                                            ║
║    Type any SQL statement ending with ;                               ║
║    .help          — show this help                                    ║
║    .tables        — list all tables                                   ║
║    .schema <tbl>  — show table schema                                 ║
║    .quit / .exit  — exit                                              ║
║    .load <file>   — load and run a .qf script file                    ║
║                                                                       ║
║  Supported SQL:                                                       ║
║    CREATE TABLE / DROP TABLE / ALTER TABLE                            ║
║    INSERT INTO … VALUES                                               ║
║    SELECT … FROM … WHERE … JOIN … GROUP BY … HAVING …                ║
║           ORDER BY … LIMIT … OFFSET                                   ║
║    UPDATE … SET … WHERE …                                             ║
║    DELETE FROM … WHERE …                                              ║
║    CREATE [UNIQUE] INDEX … ON …                                       ║
║    BEGIN / COMMIT / ROLLBACK                                          ║
║                                                                       ║
║  Expressions: +, -, *, /, %, =, !=, <, >, <=, >=                     ║
║    AND, OR, NOT, LIKE, IN, BETWEEN, IS NULL, IS NOT NULL              ║
║    CASE WHEN … THEN … ELSE … END                                     ║
║    Aggregates: COUNT, SUM, AVG, MIN, MAX                              ║
║    Scalars:    ABS, LOWER, UPPER, LENGTH, TRIM, SUBSTR, REPLACE,     ║
║                CONCAT, COALESCE, IFNULL, NULLIF, TYPEOF, ROUND       ║
║    Casts:      expr::TYPE                                             ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Pretty-print a result set as a bordered table."""
    if not headers:
        return ""
    # Stringify
    str_rows = []
    for r in rows:
        str_rows.append([("NULL" if v is None else str(v)) for v in r])
    # Column widths
    widths = [len(h) for h in headers]
    for r in str_rows:
        for i, v in enumerate(r):
            if i < len(widths):
                widths[i] = max(widths[i], len(v))
    # Build
    def line(ch="─", join="┼", left="├", right="┤"):
        return left + join.join(ch * (w + 2) for w in widths) + right
    def row_str(vals):
        cells = []
        for i, v in enumerate(vals):
            cells.append(f" {v:<{widths[i]}} ")
        return "│" + "│".join(cells) + "│"

    lines = []
    lines.append("┌" + "┬".join("─" * (w + 2) for w in widths) + "┐")
    lines.append(row_str(headers))
    lines.append(line())
    for r in str_rows:
        # pad if needed
        while len(r) < len(widths):
            r.append("")
        lines.append(row_str(r))
    lines.append("└" + "┴".join("─" * (w + 2) for w in widths) + "┘")
    return "\n".join(lines)


def run_sql(executor: Executor, sql: str) -> str:
    """Parse and execute a single SQL string.  Returns formatted output or error."""
    sql = sql.strip()
    if not sql:
        return ""
    try:
        tokens = Lexer(sql).tokenise()
        ast    = Parser(tokens).parse()
        headers, rows = executor.execute(ast)
        if headers:
            return format_table(headers, rows)
        # DDL / DML with no result
        return "OK"
    except (LexError, ParseError, StorageError, ExecError) as e:
        return f"ERROR: {e}"


def repl():
    """Interactive Read-Eval-Print Loop."""
    executor = Executor()
    print(BANNER)

    # Buffer for multi-line input
    buf = ""
    while True:
        prompt = "   > " if not buf else "   . "
        try:
            line = input(prompt)
        except EOFError:
            break

        # dot commands
        stripped = line.strip()
        if not buf and stripped.startswith("."):
            cmd = stripped.split()
            if cmd[0] in (".quit", ".exit"):
                print("Bye.")
                break
            if cmd[0] == ".help":
                print(BANNER)
                continue
            if cmd[0] == ".tables":
                tables = list(executor.catalog.tables.keys())
                print("  Tables:", ", ".join(tables) if tables else "(none)")
                continue
            if cmd[0] == ".schema":
                if len(cmd) < 2:
                    print("  Usage: .schema <table_name>")
                    continue
                try:
                    tbl = executor.catalog.get_table(cmd[1])
                    print(f"  CREATE TABLE {tbl.name} (")
                    for i, col in enumerate(tbl.columns):
                        flags = ""
                        if col.primary_key: flags += " PRIMARY KEY"
                        if col.not_null:    flags += " NOT NULL"
                        if col.unique:      flags += " UNIQUE"
                        sep = "," if i < len(tbl.columns) - 1 else ""
                        print(f"    {col.name:20s} {col.col_type:8s}{flags}{sep}")
                    print("  );")
                    if tbl.indexes:
                        print(f"  Indexes: {list(tbl.indexes.keys())}")
                except StorageError as e:
                    print(f"  ERROR: {e}")
                continue
            if cmd[0] == ".load":
                if len(cmd) < 2:
                    print("  Usage: .load <filename>")
                    continue
                try:
                    with open(cmd[1]) as f:
                        script = f.read()
                    # split on ; and run each
                    for stmt_text in script.split(";"):
                        stmt_text = stmt_text.strip()
                        if stmt_text:
                            result = run_sql(executor, stmt_text)
                            if result and result != "OK":
                                print(result)
                    print(f"  Loaded {cmd[1]}")
                except FileNotFoundError:
                    print(f"  ERROR: file '{cmd[1]}' not found")
                continue
            print(f"  Unknown command: {cmd[0]}")
            continue

        buf += line + "\n"
        # Execute when we see a semicolon
        if ";" in buf:
            # Run everything up to and including the last ;
            statements = buf.split(";")
            # last element is after the final ; (may be empty or partial)
            remainder = statements[-1]
            for stmt_text in statements[:-1]:
                stmt_text = stmt_text.strip()
                if stmt_text:
                    result = run_sql(executor, stmt_text)
                    if result:
                        print(result)
            buf = remainder


# ════════════════════════════════════════════════════════════════════════════
# 8.  DEMO  (runs automatically if no interactive terminal)
# ════════════════════════════════════════════════════════════════════════════

DEMO_SCRIPT = """
-- ═══ QueryForge Demo ═══

-- 1) Create tables
CREATE TABLE employees (
    id      INT PRIMARY KEY,
    name    TEXT NOT NULL,
    dept    TEXT NOT NULL,
    salary  REAL,
    active  BOOL
);

CREATE TABLE departments (
    dept_name TEXT PRIMARY KEY,
    budget    REAL NOT NULL,
    location  TEXT
);

-- 2) Insert data
INSERT INTO departments (dept_name, budget, location) VALUES
    ('Engineering', 500000.0, 'Building A'),
    ('Marketing',   200000.0, 'Building B'),
    ('Sales',       300000.0, 'Building C'),
    ('HR',          150000.0, 'Building A');

INSERT INTO employees (id, name, dept, salary, active) VALUES
    (1, 'Alice',   'Engineering', 120000.0, true),
    (2, 'Bob',     'Engineering',  95000.0, true),
    (3, 'Charlie', 'Marketing',    85000.0, true),
    (4, 'Diana',   'Sales',        90000.0, true),
    (5, 'Eve',     'Engineering', 110000.0, false),
    (6, 'Frank',   'HR',           75000.0, true),
    (7, 'Grace',   'Sales',        92000.0, true),
    (8, 'Hank',    'Marketing',    88000.0, false);

-- 3) Basic SELECT
SELECT * FROM employees;

-- 4) WHERE + ORDER BY + LIMIT
SELECT name, salary FROM employees WHERE salary > 90000 ORDER BY salary DESC LIMIT 3;

-- 5) GROUP BY + HAVING + aggregate
SELECT dept, COUNT(*) as cnt, AVG(salary) as avg_sal
FROM employees
GROUP BY dept
HAVING COUNT(*) > 1
ORDER BY avg_sal DESC;

-- 6) JOIN
SELECT e.name, e.salary, d.location
FROM employees e
JOIN departments d ON e.dept = d.dept_name
WHERE e.active = true
ORDER BY e.salary DESC;

-- 7) LEFT JOIN (shows NULLs for unmatched)
SELECT d.dept_name, d.budget, e.name
FROM departments d
LEFT JOIN employees e ON d.dept_name = e.dept
ORDER BY d.dept_name;

-- 8) LIKE
SELECT name FROM employees WHERE name LIKE 'A%';

-- 9) IN
SELECT name, dept FROM employees WHERE dept IN ('Engineering', 'HR');

-- 10) BETWEEN
SELECT name, salary FROM employees WHERE salary BETWEEN 80000 AND 100000;

-- 11) CASE WHEN
SELECT name, salary,
    CASE
        WHEN salary >= 110000 THEN 'Senior'
        WHEN salary >= 90000  THEN 'Mid'
        ELSE                       'Junior'
    END as level
FROM employees
ORDER BY salary DESC;

-- 12) IS NULL / IS NOT NULL
SELECT name, salary FROM employees WHERE salary IS NOT NULL;

-- 13) Scalar functions
SELECT name, UPPER(name) as upper_name, LENGTH(name) as name_len FROM employees;

-- 14) UPDATE
UPDATE employees SET salary = salary * 1.1 WHERE dept = 'Engineering';
SELECT name, salary FROM employees WHERE dept = 'Engineering';

-- 15) DELETE
DELETE FROM employees WHERE active = false;
SELECT name, active FROM employees;

-- 16) CREATE INDEX
CREATE UNIQUE INDEX idx_emp_id ON employees (id);

-- 17) ALTER TABLE — add column
ALTER TABLE employees ADD COLUMN hire_year INT;
UPDATE employees SET hire_year = 2020 WHERE id = 1;
SELECT name, hire_year FROM employees;

-- 18) DISTINCT
SELECT DISTINCT dept FROM employees ORDER BY dept;

-- 19) Subquery in WHERE
SELECT name FROM employees WHERE salary > (SELECT AVG(salary) FROM employees);

-- 20) Transaction (BEGIN / ROLLBACK)
BEGIN;
DELETE FROM employees;
SELECT COUNT(*) as cnt FROM employees;
ROLLBACK;
SELECT COUNT(*) as cnt FROM employees;
"""


def run_demo():
    """Run the demo script and print results."""
    executor = Executor()
    print(BANNER)
    print("  ─── Running Demo Script ───\n")
    step = 0
    for stmt_text in DEMO_SCRIPT.split(";"):
        stmt_text = stmt_text.strip()
        if not stmt_text:
            continue
        # Print the statement (first line as header)
        first_line = stmt_text.split("\n")[0].strip()
        if first_line.startswith("--"):
            print(f"\n  {first_line}")
            # get actual SQL (skip comment lines)
            sql_lines = [l for l in stmt_text.split("\n") if not l.strip().startswith("--")]
            stmt_text = "\n".join(sql_lines).strip()
            if not stmt_text:
                continue

        result = run_sql(executor, stmt_text)
        if result and result != "OK":
            # indent the table
            for line in result.split("\n"):
                print(f"  {line}")
        elif result == "OK":
            # Show the SQL that was run
            short = stmt_text.replace("\n", " ").strip()
            if len(short) > 70:
                short = short[:67] + "..."
            print(f"  ✓ {short}")


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    elif sys.stdin.isatty():
        repl()
    else:
        # Pipe mode: read all stdin, run as script
        script = sys.stdin.read()
        executor = Executor()
        for stmt_text in script.split(";"):
            stmt_text = stmt_text.strip()
            if stmt_text:
                result = run_sql(executor, stmt_text)
                if result:
                    print(result)
