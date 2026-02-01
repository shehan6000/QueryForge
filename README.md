# QueryForge

A relational database engine written in Python — **from scratch**. No external dependencies. No ORMs. No third-party SQL libraries. Every byte of the pipeline — lexer, parser, query planner, executor, storage, and indexing — is hand-built.

```
pip install nothing    # seriously, zero dependencies
python3 queryforge.py  # and it just works
```

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture](#architecture)
3. [Installation & Running](#installation--running)
4. [SQL Reference](#sql-reference)
   - [Data Types](#data-types)
   - [CREATE TABLE](#create-table)
   - [INSERT](#insert)
   - [SELECT](#select)
   - [UPDATE](#update)
   - [DELETE](#delete)
   - [Indexes](#indexes)
   - [ALTER TABLE](#alter-table)
   - [Transactions](#transactions)
5. [Expressions & Operators](#expressions--operators)
6. [Functions](#functions)
   - [Aggregate Functions](#aggregate-functions)
   - [Scalar Functions](#scalar-functions)
7. [REPL Commands](#repl-commands)
8. [How Each Layer Works](#how-each-layer-works)
   - [Lexer](#lexer)
   - [Parser](#parser)
   - [AST](#ast)
   - [Executor](#executor)
   - [Storage & B-Tree Indexes](#storage--b-tree-indexes)
   - [Transactions](#transaction-manager)
9. [Demo Walkthrough](#demo-walkthrough)
10. [Extending QueryForge](#extending-queryforge)

---

## What It Does

QueryForge is a fully functional SQL database engine. You can create tables, insert data, query it with joins, aggregates, subqueries, filtering, sorting, and more — all processed by a hand-written compiler pipeline inside a single Python file.

| Capability | Status |
|---|---|
| CREATE / DROP / ALTER TABLE | ✓ |
| INSERT with multi-row VALUES | ✓ |
| SELECT with WHERE, ORDER BY, LIMIT, OFFSET | ✓ |
| INNER / LEFT / RIGHT JOIN | ✓ |
| GROUP BY + HAVING | ✓ |
| Aggregate functions (COUNT, SUM, AVG, MIN, MAX) | ✓ |
| Scalar functions (12 built-in) | ✓ |
| Subqueries (scalar + EXISTS) | ✓ |
| LIKE, IN, BETWEEN, IS NULL | ✓ |
| CASE WHEN … THEN … ELSE … END | ✓ |
| DISTINCT | ✓ |
| B-Tree indexes (including UNIQUE) | ✓ |
| Transactions (BEGIN / COMMIT / ROLLBACK) | ✓ |
| Type casting with `::` | ✓ |
| Interactive REPL + pipe mode + script files | ✓ |

---

## Architecture

The engine is a classic compiler pipeline. A SQL string enters at the left and a result table exits at the right. Every stage is a discrete class — nothing is glued together with magic.

```
  ┌──────────────────────────────────────────────────────────────────┐
  │                        queryforge.py                             │
  │                                                                  │
  │  SQL text                                                        │
  │     │                                                            │
  │     ▼                                                            │
  │  ┌────────┐    token     ┌────────┐    AST      ┌────────────┐  │
  │  │ Lexer  │ ───────────▶ │ Parser │ ──────────▶ │  Executor  │  │
  │  └────────┘    stream    └────────┘   nodes     └────────────┘  │
  │                                                       │         │
  │                                                       ▼         │
  │                                              ┌──────────────┐   │
  │                                              │  Catalog     │   │
  │                                              │  ├ Table     │   │
  │                                              │  │  ├ Rows    │   │
  │                                              │  │  └ BTree   │   │
  │                                              │  └ TransMgr  │   │
  │                                              └──────────────┘   │
  └──────────────────────────────────────────────────────────────────┘
```

| Layer | Class(es) | Responsibility |
|---|---|---|
| **Lexer** | `Lexer`, `Token`, `TT` | Converts raw text into a flat stream of typed tokens. Single-pass, O(n). |
| **Parser** | `Parser` | Recursive-descent parser with full operator-precedence climbing. Produces an AST. |
| **AST** | 30+ `@dataclass` nodes | Pure data. Every possible expression and statement is its own class. |
| **Executor** | `Executor` | Walks the AST and runs it against the catalog. Handles joins, grouping, aggregation, sorting, projection. |
| **Storage** | `Catalog`, `Table`, `BTreeIndex` | In-memory row store. Each table owns its rows and its indexes. |
| **Transactions** | `TransactionManager` | Snapshot isolation via deep-copy on `BEGIN`, restore on `ROLLBACK`. |

**Total: 53 classes, ~2 300 lines, 0 dependencies.**

---

## Installation & Running

### Requirements

- Python 3.10 or later (uses `match` statements)
- Nothing else. No pip install. No virtual environment needed.

### Three ways to run it

```bash
# 1. Interactive REPL — type SQL, press Enter after the semicolon
python3 queryforge.py

# 2. Built-in demo — 20 queries showing every feature
python3 queryforge.py --demo

# 3. Pipe mode — feed SQL through stdin
echo "SELECT 1 + 1 as result;" | python3 queryforge.py

# 4. Script file — write queries to a .qf file, load in the REPL
# (see .load command below)
```

---

## SQL Reference

### Data Types

| Type | Python equivalent | Notes |
|---|---|---|
| `INT` | `int` | 64-bit integer |
| `BIGINT` | `int` | Alias, same as INT in Python |
| `REAL` | `float` | 64-bit floating point |
| `TEXT` | `str` | Variable-length string |
| `BLOB` | `bytes` | Binary data |
| `BOOL` | `bool` | `true` / `false` |

---

### CREATE TABLE

```sql
CREATE TABLE employees (
    id      INT    PRIMARY KEY,
    name    TEXT   NOT NULL,
    dept    TEXT   NOT NULL,
    salary  REAL,
    active  BOOL
);
```

Supported column constraints: `PRIMARY KEY`, `NOT NULL`, `UNIQUE`.

Supported table constraints: `PRIMARY KEY (col1, col2)`, `FOREIGN KEY … REFERENCES …`.

Use `IF NOT EXISTS` to suppress the error when the table already exists:

```sql
CREATE TABLE IF NOT EXISTS logs (id INT, msg TEXT);
```

---

### INSERT

```sql
-- Specify columns explicitly
INSERT INTO employees (id, name, dept, salary, active)
VALUES (1, 'Alice', 'Engineering', 120000.0, true);

-- Multi-row insert
INSERT INTO employees (id, name, dept, salary, active) VALUES
    (2, 'Bob',     'Engineering', 95000.0, true),
    (3, 'Charlie', 'Marketing',   85000.0, true);

-- Omit column list to insert in definition order
INSERT INTO employees VALUES (4, 'Diana', 'Sales', 90000.0, true);
```

---

### SELECT

Full syntax (every clause is optional except `SELECT`):

```sql
SELECT [DISTINCT] columns
FROM   table_or_join
WHERE  condition
GROUP BY expressions
HAVING condition
ORDER BY expressions [ASC | DESC]
LIMIT  n
OFFSET n
```

**Basic queries:**

```sql
SELECT * FROM employees;

SELECT name, salary FROM employees WHERE salary > 90000;

SELECT name, salary
FROM   employees
ORDER BY salary DESC
LIMIT  3;
```

**Aliases:**

```sql
SELECT name AS employee_name, salary * 1.1 AS projected
FROM employees e                          -- table alias
WHERE e.dept = 'Engineering';
```

**JOINs:**

```sql
-- INNER JOIN
SELECT e.name, d.location
FROM employees e
JOIN departments d ON e.dept = d.dept_name;

-- LEFT JOIN (unmatched right rows become NULL)
SELECT d.dept_name, e.name
FROM departments d
LEFT JOIN employees e ON d.dept_name = e.dept;

-- RIGHT JOIN
SELECT e.name, d.dept_name
FROM employees e
RIGHT JOIN departments d ON e.dept = d.dept_name;
```

**Chained joins:**

```sql
SELECT a.name, b.name, c.name
FROM   t1 a
JOIN   t2 b ON a.id = b.t1_id
JOIN   t3 c ON b.id = c.t2_id;
```

**GROUP BY + HAVING:**

```sql
SELECT dept, COUNT(*) AS cnt, AVG(salary) AS avg_sal
FROM   employees
GROUP BY dept
HAVING COUNT(*) > 1
ORDER BY avg_sal DESC;
```

**Subqueries:**

```sql
-- Scalar subquery in WHERE
SELECT name FROM employees
WHERE  salary > (SELECT AVG(salary) FROM employees);

-- EXISTS
SELECT name FROM employees e
WHERE  EXISTS (
    SELECT 1 FROM departments d
    WHERE  d.dept_name = e.dept AND d.budget > 200000
);

-- IN with subquery
SELECT name FROM employees
WHERE  dept IN (SELECT dept_name FROM departments WHERE budget > 200000);
```

---

### UPDATE

```sql
UPDATE employees
SET    salary = salary * 1.1
WHERE  dept = 'Engineering';

-- Multiple columns
UPDATE employees
SET    salary = 100000, active = true
WHERE  id = 5;
```

---

### DELETE

```sql
DELETE FROM employees WHERE active = false;

-- Delete all rows (no WHERE)
DELETE FROM employees;
```

---

### Indexes

```sql
-- Regular index
CREATE INDEX idx_dept ON employees (dept);

-- Unique index (enforced on INSERT and UPDATE)
CREATE UNIQUE INDEX idx_emp_id ON employees (id);

-- Composite index
CREATE INDEX idx_dept_salary ON employees (dept, salary);

-- Drop
DROP INDEX idx_dept;
DROP INDEX IF EXISTS idx_dept;   -- no error if missing
```

Indexes are automatically maintained on every INSERT, UPDATE, and DELETE. Primary key columns get a unique index automatically.

---

### ALTER TABLE

```sql
-- Add a column
ALTER TABLE employees ADD COLUMN hire_year INT;
ALTER TABLE employees ADD hire_year INT;     -- COLUMN keyword is optional

-- Drop a column
ALTER TABLE employees DROP COLUMN hire_year;

-- Rename a column
ALTER TABLE employees RENAME hire_year TO start_year;

-- Rename the table itself
ALTER TABLE employees RENAME TO staff;
```

---

### Transactions

```sql
BEGIN;

INSERT INTO employees VALUES (9, 'Zara', 'HR', 70000.0, true);
DELETE FROM employees WHERE id = 1;

-- Save everything above
COMMIT;
```

```sql
BEGIN;

DELETE FROM employees;          -- oops

-- Undo everything back to the BEGIN
ROLLBACK;

SELECT COUNT(*) FROM employees; -- rows are back
```

QueryForge uses **snapshot isolation**: `BEGIN` takes a deep copy of the entire database state. `ROLLBACK` restores from that copy. `COMMIT` discards the copy (changes are already live).

---

## Expressions & Operators

| Operator | Example | Notes |
|---|---|---|
| Arithmetic | `salary * 1.1 + 1000` | `+` `-` `*` `/` `%` |
| String concat | `first_name + ' ' + last_name` | `+` on strings concatenates |
| Comparison | `salary > 50000` | `=` `!=` `<` `>` `<=` `>=` |
| Logic | `active = true AND salary > 80000` | `AND` `OR` `NOT` |
| LIKE | `name LIKE 'A%'` | `%` = any sequence, `_` = any single char. Case-insensitive. |
| NOT LIKE | `name NOT LIKE '%x%'` | |
| IN | `dept IN ('HR', 'Sales')` | |
| NOT IN | `dept NOT IN ('HR')` | |
| IN subquery | `dept IN (SELECT …)` | |
| BETWEEN | `salary BETWEEN 80000 AND 100000` | Inclusive on both ends |
| NOT BETWEEN | `salary NOT BETWEEN 80000 AND 100000` | |
| IS NULL | `salary IS NULL` | |
| IS NOT NULL | `salary IS NOT NULL` | |
| CASE | see below | |
| Cast | `price::INT` | Converts value to the target type |
| Grouping | `(a + b) * c` | Parentheses for precedence |

**Operator precedence** (lowest → highest):

```
OR
AND
NOT
=  !=  <  >  <=  >=  LIKE  IN  BETWEEN  IS NULL
+  -
*  /  %
unary -  unary NOT
( )  literals  column refs  function calls
```

**CASE expressions:**

```sql
-- Searched CASE (no operand)
SELECT name,
    CASE
        WHEN salary >= 110000 THEN 'Senior'
        WHEN salary >= 90000  THEN 'Mid'
        ELSE                       'Junior'
    END AS level
FROM employees;

-- Simple CASE (with operand)
SELECT name,
    CASE dept
        WHEN 'Engineering' THEN 'Eng'
        WHEN 'Marketing'   THEN 'Mkt'
        ELSE                    'Other'
    END AS dept_short
FROM employees;
```

---

## Functions

### Aggregate Functions

Aggregate functions collapse a group of rows into a single value. Use them in `SELECT` with or without `GROUP BY`. All of them ignore `NULL` values (except `COUNT(*)`).

| Function | Example | Returns |
|---|---|---|
| `COUNT(*)` | `COUNT(*)` | Number of rows in the group |
| `COUNT(col)` | `COUNT(salary)` | Number of non-NULL values |
| `COUNT(DISTINCT col)` | `COUNT(DISTINCT dept)` | Number of distinct non-NULL values |
| `SUM(col)` | `SUM(salary)` | Sum of non-NULL values, or NULL if all are NULL |
| `AVG(col)` | `AVG(salary)` | Arithmetic mean of non-NULL values |
| `MIN(col)` | `MIN(salary)` | Smallest non-NULL value |
| `MAX(col)` | `MAX(salary)` | Largest non-NULL value |

`DISTINCT` is supported inside any aggregate:

```sql
SELECT COUNT(DISTINCT dept) AS unique_depts FROM employees;
SELECT SUM(DISTINCT salary) AS unique_salary_total FROM employees;
```

---

### Scalar Functions

Scalar functions operate on a single value per row.

| Function | Signature | Description |
|---|---|---|
| `ABS` | `ABS(x)` | Absolute value |
| `LOWER` | `LOWER(s)` | Lowercase string |
| `UPPER` | `UPPER(s)` | Uppercase string |
| `LENGTH` | `LENGTH(s)` | Number of characters |
| `TRIM` | `TRIM(s)` | Strip leading and trailing whitespace |
| `SUBSTR` | `SUBSTR(s, start)` or `SUBSTR(s, start, len)` | Substring. `start` is 1-based (SQL convention). |
| `REPLACE` | `REPLACE(s, old, new)` | Replace all occurrences |
| `CONCAT` | `CONCAT(a, b, …)` | Concatenate any number of values |
| `COALESCE` | `COALESCE(a, b, …)` | First non-NULL argument |
| `IFNULL` | `IFNULL(a, b)` | Returns `a` if not NULL, else `b`. Alias: `NVL`. |
| `NULLIF` | `NULLIF(a, b)` | Returns NULL if `a = b`, else `a` |
| `TYPEOF` | `TYPEOF(x)` | Returns the type as a string: `INT`, `REAL`, `TEXT`, `BOOL`, `NULL` |
| `ROUND` | `ROUND(x)` or `ROUND(x, n)` | Round to `n` decimal places (default 0) |

All scalar functions return `NULL` if any required input is `NULL` (NULL propagation), except `COALESCE` and `IFNULL` which are specifically designed to handle NULLs.

---

## REPL Commands

When running in interactive mode (`python3 queryforge.py`), these dot-commands are available:

| Command | Description |
|---|---|
| `.help` | Print the full help banner |
| `.tables` | List every table in the database |
| `.schema tablename` | Print the CREATE TABLE statement and index list for a table |
| `.load filename` | Read a `.qf` script file and execute every statement in it |
| `.quit` or `.exit` | Exit the REPL |

Multi-line input is supported. The REPL buffers input until it sees a `;`:

```
   > SELECT name,
   .        salary
   .   FROM employees
   .  WHERE salary > 90000;
```

---

## How Each Layer Works

### Lexer

**Class:** `Lexer` — **Lines:** 105–260

The lexer is a hand-written, single-pass character scanner. It does not use regular expressions on the hot path. It processes the input byte-by-byte, recognising:

- **Keywords** via a hash-map lookup on the lowercased identifier text. This means the language is case-insensitive for keywords (`SELECT` = `select` = `Select`).
- **String literals** delimited by single quotes. Escaped quotes use SQL doubling: `'it''s'` → `it's`.
- **Numbers** including negative literals, decimals, and scientific notation (`1.5e10`).
- **Comments** — both single-line (`--`) and block (`/* … */`), which are simply skipped.
- **Operators** — one-char and two-char (`!=`, `<=`, `>=`, `::`), distinguished by one character of lookahead.

Output: a `Vec` of `Token` objects, each carrying its type, value (if any), and source position (line, col) for error reporting.

---

### Parser

**Class:** `Parser` — **Lines:** 474–1010

A recursive-descent parser with **precedence climbing** for expressions. Each precedence level is its own method:

```
_expr → _or → _and → _not → _comparison → _addition → _multiply → _unary → _atom
```

Each method calls the one to its right (higher precedence), loops on its own operator, and builds a `BinOp` node. This makes operator precedence correct by construction — no precedence table, no Pratt parsing, just the call stack.

Statement-level parsing is a simple `match` on the first token: `CREATE` dispatches to `_create_table` or `_create_index`, `SELECT` to `_select`, etc.

Sub-selects are handled naturally: whenever the parser sees `(SELECT …)` inside an expression, it recurses into `_select` and wraps the result in a `SubSelectExpr` node.

---

### AST

**Nodes:** 30+ `@dataclass` classes — **Lines:** 262–461

The AST is pure data — no methods, no behaviour. Every expression type is its own class (`LitInt`, `BinOp`, `FuncCall`, `CaseExpr`, `InExpr`, etc.) and every statement type is its own class (`SelectStmt`, `InsertStmt`, `CreateTableStmt`, etc.).

This makes the executor's job straightforward: a single `match` statement on the node type dispatches to the correct evaluation logic with full type safety. No isinstance chains, no visitor pattern — just pattern matching.

---

### Executor

**Class:** `Executor` — **Lines:** 1218–1944

The executor is the largest layer. It walks the AST and produces results. The key methods:

| Method | What it does |
|---|---|
| `_eval_from` | Scans one or more tables, produces a list of row-dicts |
| `_do_join` | Nested-loop join. Handles INNER, LEFT, RIGHT. |
| `_group_and_aggregate` | Groups rows by key, attaches the group list to each output row so aggregate functions can iterate over it |
| `_project` | Evaluates the SELECT column list against each row, expands `*` |
| `_eval_expr` | Recursive expression evaluator. One big `match` on AST node type. Handles NULL propagation, short-circuit AND/OR, LIKE-to-regex conversion. |
| `_eval_func` | Dispatches to aggregate or scalar function implementations |

**NULL semantics** follow SQL convention: any arithmetic or comparison involving NULL returns NULL. `AND`/`OR` short-circuit. `COUNT(*)` counts rows including NULLs; `COUNT(col)` skips them.

**LIKE** is implemented by converting the SQL pattern (`%` and `_`) into a Python regex at evaluation time, then matching case-insensitively.

---

### Storage & B-Tree Indexes

**Classes:** `Table`, `BTreeIndex`, `Catalog` — **Lines:** 1018–1150

Each table stores its rows in a `dict[int, list]` mapping row-id to the list of column values. Row-ids are monotonically increasing integers, auto-assigned on INSERT.

`BTreeIndex` is a sorted-list structure using binary search for O(log n) lookups. It maps a tuple of indexed column values to a list of row-ids. On INSERT, the new key is placed at its sorted position. On DELETE, the key is found and the row-id removed. UNIQUE indexes raise an error if a duplicate key is inserted.

Every table can hold multiple indexes simultaneously. All indexes are updated automatically on every INSERT, UPDATE, and DELETE — the `Table` class coordinates this internally so the executor never has to think about it.

`Catalog` is the top-level registry: a dict of table name → `Table`. It is the single source of truth for the entire database.

---

### Transaction Manager

**Class:** `TransactionManager` — **Lines:** 1156–1187

Transactions use **snapshot isolation** implemented via Python's `copy.deepcopy`:

1. **BEGIN** — the entire `Catalog` (all tables, all rows, all indexes) is deep-copied and stored as a snapshot.
2. All subsequent operations mutate the live catalog as normal.
3. **COMMIT** — the snapshot is discarded. Changes are permanent.
4. **ROLLBACK** — the live catalog's state is replaced with the snapshot. All changes since BEGIN vanish.

This is simple, correct, and gives full isolation. The trade-off is memory: the snapshot doubles the memory footprint for the duration of the transaction. For a production system you would replace this with a WAL (write-ahead log), but for correctness and clarity this approach is ideal.

---

## Demo Walkthrough

Run `python3 queryforge.py --demo` to see all of the following execute live:

| # | What it demonstrates |
|---|---|
| 1 | `CREATE TABLE` with PRIMARY KEY, NOT NULL |
| 2 | Multi-row `INSERT INTO … VALUES` |
| 3 | `SELECT *` — full table scan |
| 4 | `WHERE` + `ORDER BY DESC` + `LIMIT` |
| 5 | `GROUP BY` + `HAVING` + `COUNT` + `AVG` |
| 6 | `INNER JOIN` with table aliases |
| 7 | `LEFT JOIN` — NULL-filled unmatched rows |
| 8 | `LIKE 'A%'` pattern matching |
| 9 | `IN ('Engineering', 'HR')` list membership |
| 10 | `BETWEEN 80000 AND 100000` range check |
| 11 | `CASE WHEN … THEN … ELSE … END` |
| 12 | `IS NOT NULL` filter |
| 13 | Scalar functions: `UPPER`, `LENGTH` |
| 14 | `UPDATE … SET … WHERE` with arithmetic expression |
| 15 | `DELETE … WHERE` |
| 16 | `CREATE UNIQUE INDEX` |
| 17 | `ALTER TABLE … ADD COLUMN` + verify with SELECT |
| 18 | `SELECT DISTINCT` |
| 19 | Scalar subquery in `WHERE` clause |
| 20 | `BEGIN` → `DELETE` → `ROLLBACK` — full transaction round-trip |

---

## Extending QueryForge

QueryForge is designed to be easy to extend. Here are the three most common additions and exactly where to make them:

**Adding a new scalar function** — open `queryforge.py`, find the `_eval_func` method in the `Executor` class, and add a new `if NAME == "..."` block before the aggregate section. Example:

```python
if NAME == "REVERSE":
    v = self._eval_expr(args[0], row)
    return None if v is None else str(v)[::-1]
```

**Adding a new statement type** — three steps: (1) add a new `@dataclass` in the AST section, (2) add a parsing method in the `Parser` class and wire it into `_statement`, (3) add an execution method in `Executor` and wire it into `execute`.

**Adding a new data type** — add the keyword to `TT` and `KEYWORDS` in the lexer section, add it to the `TYPE_MAP` dicts in the parser's `_column_def` and `_maybe_cast` methods, and add a coercion case in the `_coerce` function.

---

