
#[derive(Debug, Clone, PartialEq)]
pub enum ConstLiteral {
    IntegerLiteral(i64),
    StringLiteral(String),
    BoolLiteral(bool),
    CharLiteral(char)
}

pub struct Identifier {
    name: String
}

pub struct Arguments {
    args: Vec<Expr>,
    vararg: Option<Identifier>,
    kwargs: Option<Identifier>,
    defaults: Vec<Expr>
}

pub enum Stmt {
    Func {
        name: Box<Identifier>,
        args: Box<Arguments>,
        body: BlockStmt
    },
    Return {
        value: Option<Expr>
    },
    While {
        cond: Box<Expr>,
        block: BlockStmt
    },
    If {
        cond: Box<Expr>,
        body: BlockStmt,
        or_else: Option<BlockStmt>
    },
    Expr {
        expr: Box<Expr>
    },
    Assign {
        targets: Vec<Expr>,
        value: Expr
    },
    Break,
    Continue,
}

pub type BlockStmt = Vec<Stmt>;

pub enum Expr {
    LiteralExpr {
        value: ConstLiteral
    },
    BoolOp {
        bop: Bop,
        values: Vec<Expr>
    },
    BinOp {
        left: Box<Expr>,
        op: Op,
        right: Box<Expr>
    },
    UnaryOp {
        op: Uop,
        exp: Box<Expr>
    },
    Name {
        name: Identifier
    }
}

pub enum Bop {
    And,
    Or
}

pub enum Uop {
    Not,
    UAdd,
    USub,
}

pub enum Op {
    Add,
    Mins,
    Mult,
    Div,
    Mod,

    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE
}
