
#[derive(Debug, Clone, PartialEq)]
pub enum ConstLiteral {
    IntegerLiteral(i64),
    StringLiteral(String),
    FloatLiteral(f64),
    BoolLiteral(bool),
    CharLiteral(char)
}

#[derive(PartialEq, Clone, Debug)]
pub struct Identifier {
    pub name: String
}

#[derive(Clone, Debug, PartialEq)]
pub struct Arguments {
    pub args: Vec<Identifier>
}

#[derive(Clone, Debug, PartialEq)]
pub enum Stmt {
    Return {
        value: Option<Expr>
    },
    While {
        cond: Expr,
        block: BlockStmt
    },
    Expr {
        expr: Expr
    },
    Assign {
        ident: Identifier,
        value: Expr
    },
    Break,
    Continue,
}

pub type BlockStmt = Vec<Stmt>;

#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    LiteralExpr {
        value: ConstLiteral
    },
    InfixExpr {
        left: Box<Expr>,
        op: Op,
        right: Box<Expr>
    },
    PrefixExpr {
        op: Uop,
        expr: Box<Expr>
    },
    Name {
        name: Identifier
    },
    FnCallExpr {
        fun: Box<Expr>,
        params: Vec<Expr>
    },
    ArrayExpr {
        values: Vec<Expr>
    },
    ArrayIndexExpr {
        array: Box<Expr>,
        index: Box<Expr>
    },
    If {
        cond: Box<Expr>,
        body: BlockStmt,
        or_else: Option<BlockStmt>
    },
    Func {
        name: Identifier,
        args: Arguments,
        body: BlockStmt
    }
}

#[derive(PartialOrd, PartialEq, Clone, Debug)]
pub enum Bop {
    And,
    Or
}

#[derive(PartialOrd, PartialEq, Clone, Debug)]
pub enum Uop {
    Not,
    UAdd,
    USub,
}

#[derive(PartialOrd, PartialEq, Clone, Debug)]
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

#[derive(PartialEq, PartialOrd, Debug, Clone)]
pub enum Precedence {
    PLowest,
    PEquals,
    PLessGreater,
    PSum,
    PProduct,
    PCall,
    PIndex,
}

pub type Program = BlockStmt;
