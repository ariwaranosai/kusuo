use nom::*;

use lexer::*;
use ast::*;
use ast::ConstLiteral::*;

macro_rules! tag_token (
    ($i: expr, $tag: expr) => ({
        let (i1, t1) = try_parse!($i, take!(1));
        if t1.toks.is_empty() {
            IResult::Incomplete::<_,_,u32>(Needed::Size(1))
        } else {
            if t1.toks[0] == $tag {
               IResult::Done(i1, t1)
            } else {
               IResult::Error(error_position!(ErrorKind::Count, $i))
            }
        }
    })
);

macro_rules! parse_ident (
    ($i: expr,) => ({
        let (i1, t1) = try_parse!($i, take!(1));
        if t1.toks.is_empty() {
            IResult::Error(error_position!(ErrorKind::Tag, $i))
        } else {
            match t1.toks[0].clone() {
                Token::Identifier(name) =>
                    IResult::Done(i1,
                        Identifier {
                            name
                        }),
                _ => IResult::Error(error_position!(ErrorKind::Tag, $i)),
            }
        }
    });
);

macro_rules! parse_literal (
    ($i: expr, ) => ({
        let (i1, t1) = try_parse!($i, take!(1));
        if t1.toks.is_empty() {
            IResult::Error(error_position!(ErrorKind::Tag, $i))
        } else {
            match t1.toks[0].clone() {
                Token::Integer(i) => IResult::Done(i1, IntegerLiteral(i)),
                Token::Bool(b) => IResult::Done(i1, BoolLiteral(b)),
                Token::String(s) => IResult::Done(i1, StringLiteral(s)),
                Token::Char(c) => IResult::Done(i1, CharLiteral(c)),
                Token::Float(f) => IResult::Done(i1, FloatLiteral(f)),
                _ => IResult::Error(error_position!(ErrorKind::Tag, $i))
            }
        }
    });
);

named!(parse_expr<Tokens, Expr>,
    apply!(parse_pratt_expr, Precedence::PLowest)
);

named!(parse_stmt<Tokens, Stmt>,
     alt_complete!(
        parse_return_stmt |
        parse_assign_stmt |
        parse_loop_token |
        parse_expr_stmt)

);

named!(parse_stmt_end<Tokens, ()>, do_parse!(
    many0!(alt_complete!(
            preceded!(tag_token!(Token::Symbol(Symbol::SemiColon)),
             tag_token!(Token::Symbol(Symbol::LineEnd))) |
            tag_token!(Token::Symbol(Symbol::SemiColon)) |
            tag_token!(Token::Symbol(Symbol::LineEnd))
    )) >> (())
));

named!(parse_opt_end<Tokens, ()>, do_parse!(
        many0!(tag_token!(Token::Symbol(Symbol::LineEnd))) >> ()
   )
);

named!(parse_return_stmt<Tokens, Stmt>,
    do_parse!(
        tag_token!(Token::Keyword(Keyword::Return)) >>
        expr: opt!(parse_expr) >>
        parse_stmt_end >>
        (Stmt::Return {
            value: expr
        })

    )
);

named!(parse_expr_stmt<Tokens, Stmt>,
    do_parse!(
        expr: parse_expr >>
        parse_stmt_end >>
        (Stmt::Expr {expr})
    )
);

fn parse_loop_token(input: Tokens) ->  IResult<Tokens, Stmt> {
    let (i1, t1) = try_parse!(input,
                        alt_complete!(
                            tag_token!(Token::Keyword(Keyword::Break)) |
                            tag_token!(Token::Keyword(Keyword::Continue))));
    if t1.toks.is_empty() {
        IResult::Error(error_position!(ErrorKind::Tag, input))
    } else {
        match t1.toks[0].clone() {
            Token::Keyword(Keyword::Continue) => IResult::Done(i1, Stmt::Continue),
            Token::Keyword(Keyword::Break) => IResult::Done(i1, Stmt::Break),
            _ => IResult::Error(Err::Code(ErrorKind::Custom(22)))
        }
    }
}

//
fn parse_pratt_expr(input: Tokens, precedence: Precedence) -> IResult<Tokens, Expr> {
    do_parse!(input,
        left: parse_atom_expr >>
        i: apply!(go_parse_pratt_expr, precedence, left) >>
        (i)
    )
}

fn go_parse_pratt_expr(input: Tokens, precedence: Precedence, left: Expr) -> IResult<Tokens, Expr> {
    let (i1, t1) = try_parse!(input, take!(1));
    if t1.toks.is_empty() {
        IResult::Done(i1, left)
    } else {
        let peek_one = t1.toks[0].clone();
        match op_token(&peek_one) {
            (Precedence::PCall, _) if precedence < Precedence::PCall => {
                let (i2, left2) = try_parse!(input, apply!(parse_call_expr, left));
                go_parse_pratt_expr(i2, precedence, left2)
            }
            (Precedence::PIndex, _) if precedence < Precedence::PIndex => {
                let (i2, left2) = try_parse!(input, apply!(parse_index_expr, left));
                go_parse_pratt_expr(i2, precedence, left2)
            }
            (ref peek_precedunce, _) if precedence < *peek_precedunce => {
                let (i2, left2) = try_parse!(input, apply!(parse_infix_expr, left));
                go_parse_pratt_expr(i2, precedence, left2)
            }
            _ => IResult::Done(input, left),
        }
    }
}

fn parse_infix_expr(input: Tokens, left: Expr) -> IResult<Tokens, Expr> {
    let (i1, t1) = try_parse!(input, take!(1));
    if t1.toks.is_empty() {
        IResult::Error(error_position!(ErrorKind::Tag, input))
    } else {
        let next = t1.toks[0].clone();
        let (precedence, maybe_op) = op_token(&next);
        match maybe_op {
            None => IResult::Error(error_position!(ErrorKind::Tag, input)),
            Some(op) => {
                let (i2, right) = try_parse!(i1, apply!(parse_pratt_expr, precedence));
                IResult::Done(i2, Expr::InfixExpr {op, left: Box::new(left), right: Box::new(right)})
            }
        }
    }
}

named!(parse_array_expr<Tokens, Expr>,
    do_parse!(
        tag_token!(Token::Symbol(Symbol::LBracket)) >>
        values: alt_complete!(parse_exprs | empty_boxed_vec) >>
        tag_token!(Token::Symbol(Symbol::RBracket)) >>
        (Expr::ArrayExpr {
            values
        })
    )
);

fn parse_index_expr(input: Tokens, arr: Expr) -> IResult<Tokens, Expr> {
    do_parse!(
        input,
        tag_token!(Token::Symbol(Symbol::LBracket)) >>
        idx: parse_expr >>
        tag_token!(Token::Symbol(Symbol::LBracket)) >>
        (Expr::ArrayIndexExpr {
             array: Box::new(arr),
             index: Box::new(idx),
        })
    )
}

fn op_token(t: &Token) -> (Precedence, Option<Op>) {
    match *t {
        Token::Symbol(Symbol::Plus) =>
            (Precedence::PSum, Some(Op::Add)),
        Token::Symbol(Symbol::Minus) =>
            (Precedence::PSum, Some(Op::Mins)),
        Token::Symbol(Symbol::Mult) =>
            (Precedence::PProduct, Some(Op::Mult)),
        Token::Symbol(Symbol::Div) =>
            (Precedence::PProduct, Some(Op::Div)),
        Token::Symbol(Symbol::LParenthesis) =>
            (Precedence::PCall, None),
        Token::Symbol(Symbol::LBrace) =>
            (Precedence::PIndex, None),
        Token::Symbol(Symbol::Equal) =>
            (Precedence::PEquals, Some(Op::Eq)),
        Token::Symbol(Symbol::NotEqual) =>
            (Precedence::PEquals, Some(Op::NotEq)),
        Token::Symbol(Symbol::LE) =>
            (Precedence::PLessGreater, Some(Op::LtE)),
        Token::Symbol(Symbol::LT) =>
            (Precedence::PLessGreater, Some(Op::Lt)),
        Token::Symbol(Symbol::GT) =>
            (Precedence::PLessGreater, Some(Op::Gt)),
        Token::Symbol(Symbol::GE) =>
            (Precedence::PLessGreater, Some(Op::GtE)),
        _ => (Precedence::PLowest, None)
    }
}


fn parse_prefix_expr(input: Tokens) -> IResult<Tokens, Expr> {
    let (i1, t1) = try_parse!(
                       input,
                       alt_complete!(
                           tag_token!(Token::Symbol(Symbol::Plus)) |
                           tag_token!(Token::Symbol(Symbol::Minus)) |
                           tag_token!(Token::Symbol(Symbol::Not))
                       )
                   );

    if t1.toks.is_empty() {
        IResult::Error(error_position!(ErrorKind::Tag, input))
    } else {
        let (i2, e) = try_parse!(i1, parse_atom_expr);

        match t1.toks[0].clone() {
            Token::Symbol(Symbol::Plus) => IResult::Done(i2, Expr::PrefixExpr {
                op: Uop::UAdd,
                expr: Box::new(e)
            }),
            Token::Symbol(Symbol::Minus) => IResult::Done(i2, Expr::PrefixExpr {
                op: Uop::USub,
                expr: Box::new(e)
            }),
            Token::Symbol(Symbol::Not) => IResult::Done(i2, Expr::PrefixExpr {
                op: Uop::Not,
                expr: Box::new(e)
            }),
            _ => IResult::Error(Err::Code(ErrorKind::Custom(22))),
        }
    }
}

named!(parse_atom_expr<Tokens, Expr>, alt_complete!(
       parse_lit_expr |
     parse_ident_expr |
    parse_prefix_expr |
     parse_paren_expr |
     parse_array_expr |
        parse_if_expr |
        parse_fn_expr
));

named!(parse_ident_expr<Tokens, Expr>,
    do_parse!(
        ident: parse_ident!() >>
        (Expr::Name {
            name: ident
        })
    )
);

named!(parse_paren_expr<Tokens, Expr>,
    do_parse!(
        tag_token!(Token::Symbol(Symbol::LParenthesis)) >>
        expr: parse_expr >>
        tag_token!(Token::Symbol(Symbol::RParenthesis)) >>
        (expr)
    )
);

named!(parse_assign_stmt<Tokens, Stmt>,
    do_parse!(
        ident: parse_ident!() >>
        tag_token!(Token::Symbol(Symbol::Assign)) >>
        expr: parse_expr >>
        parse_stmt_end >>
        (Stmt::Assign {
            ident,
            value: expr
        })
    )
);

// literal
named!(parse_lit_expr<Tokens, Expr>,
    do_parse!(
        lit: parse_literal!() >>
        (Expr::LiteralExpr{
            value: lit
        })
    )
);

// if
named!(parse_if_expr<Tokens, Expr>,
    do_parse!(
        tag_token!(Token::Keyword(Keyword::If)) >>
        tag_token!(Token::Symbol(Symbol::LParenthesis)) >>
        expr: parse_expr >>
        tag_token!(Token::Symbol(Symbol::RParenthesis)) >>
        b: parse_block_stmt >>
        e: parse_else_expr >>
        (Expr::If{
            cond: Box::new(expr),
            body: b,
            or_else: e
        })
    )
);

named!(parse_else_expr<Tokens, Option<BlockStmt>>,
    opt!(do_parse!(
        tag_token!(Token::Keyword(Keyword::Else)) >>
        b: parse_block_stmt
        >> (b)
    ))
);

named!(parse_block_stmt<Tokens, BlockStmt>,
    do_parse!(
        tag_token!(Token::Symbol(Symbol::LBrace)) >>
        parse_opt_end >>
        ss: many0!(parse_stmt) >>
        tag_token!(Token::Symbol(Symbol::RBrace)) >>
        (ss)
    )
);

named!(parse_fn_expr<Tokens, Expr>,
    do_parse!(
        tag_token!(Token::Keyword(Keyword::Def)) >>
        name: parse_ident!() >>
        tag_token!(Token::Symbol(Symbol::LParenthesis)) >>
        parse_opt_end >>
        params: alt_complete!(parse_params | empty_params) >>
        tag_token!(Token::Symbol(Symbol::LParenthesis)) >>
        block: parse_block_stmt >>
        (Expr::Func {
            name: name,
            args: params,
            body: block
        })
    )
);

named!(parse_comma_exprs<Tokens, Expr>,
    do_parse!(
        tag_token!(Token::Symbol(Symbol::Comma)) >>
        e: parse_expr >>
        (e)
    )
);

named!(parse_exprs<Tokens, Vec<Expr>>,
    do_parse!(
        e: parse_expr >>
        es: many0!(parse_comma_exprs) >>
        ([&vec!(e)[..], &es[..]].concat())
    )
);

fn empty_boxed_vec(i: Tokens) -> IResult<Tokens, Vec<Expr>> {
    IResult::Done(i, vec![])
}

fn empty_params(i: Tokens) -> IResult<Tokens, Arguments> {
    IResult::Done(i, Arguments {
        args: vec![]
    })
}

fn parse_call_expr(input: Tokens, fn_expr: Expr) -> IResult<Tokens, Expr> {
    do_parse!(input,
        tag_token!(Token::Symbol(Symbol::LParenthesis)) >>
        args: alt_complete!(parse_exprs | empty_boxed_vec) >>
        tag_token!(Token::Symbol(Symbol::LParenthesis)) >>
        (Expr::FnCallExpr {
                fun: Box::new(fn_expr),
                params: args
        })
    )
}

named!(parse_params<Tokens, Arguments>,
    do_parse!(
        p: parse_ident!() >>
        ps: many0!(do_parse!(
            tag_token!(Token::Symbol(Symbol::Comma)) >>
            i: parse_ident!() >> (i)
        )) >> (parse_args(p, ps))
    )
);

fn parse_args(p: Identifier, ps: Vec<Identifier>) -> Arguments {
    Arguments {
        args: [&vec!(p)[..], &ps[..]].concat()
    }
}

macro_rules! parse_ident {
   ($i: expr, ) => ({
       let (i1, t1) = try_parse!($i, take!(1));
       if t1.toks.is_empty() {
           IResult::Error(error_position!(ErrorKind::Tag, $i))
       } else {
           match t1.toks[0].clone() {
               Token::Identifier(name) => IResult::Done(i1, Identifier { name })
               _ => IResult::Error(error_position!(ErrorKind::Tag, $i)),
           }
       }
   });
}

named!(parse_program<Tokens, Program>,
    do_parse!(
        programs: many0!(parse_stmt) >>
        tag_token!(Token::EOF) >>
        (programs)
    )
);

pub struct Parser;

impl Parser {
    pub fn parse_tokens(tokens: Tokens) -> IResult<Tokens, Program> {
        parse_program(tokens)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use lexer::*;

    fn assert_input_with_program(input: &[u8], expect: Program) {
        let r = Lexer::lex_tokens(input).to_result().unwrap();
        let tokens = Tokens::new(&r);
        let result = Parser::parse_tokens(tokens).to_result().unwrap();
        println!("{:?}", result);
        assert_eq!(result, expect);
    }

    fn is_the_same_inputs(input1: &[u8], input2: &[u8]) {
        let r = Lexer::lex_tokens(input1).to_result().unwrap();
        let tokens = Tokens::new(&r);
        let result1 = Parser::parse_tokens(tokens).to_result().unwrap();

        let r = Lexer::lex_tokens(input2).to_result().unwrap();
        let tokens = Tokens::new(&r);
        let result2 = Parser::parse_tokens(tokens).to_result().unwrap();

        assert_eq!(result1, result2);
    }

    #[test]
    fn assign_stmts() {
        let input = b"x = 5;
        y = 12
        z = true;
        k = 12.3; k = 12.5e3
        f = a + b
        ;;
        ";
        let result = vec![
            Stmt::Assign { ident: Identifier { name: "x".to_owned() }, value: Expr::LiteralExpr { value: IntegerLiteral(5)}},
            Stmt::Assign { ident: Identifier { name: "y".to_owned() }, value: Expr::LiteralExpr { value: IntegerLiteral(12) } },
            Stmt::Assign { ident: Identifier { name: "z".to_owned() }, value: Expr::LiteralExpr { value: BoolLiteral(true) } },
            Stmt::Assign { ident: Identifier { name: "k".to_owned() }, value: Expr::LiteralExpr { value: FloatLiteral(12.3) } },
            Stmt::Assign { ident: Identifier { name: "k".to_owned() }, value: Expr::LiteralExpr { value: FloatLiteral(12.5e3) } },
            Stmt::Assign { ident: Identifier { name: "f".to_owned() }, value: Expr::InfixExpr {
                                                                                    right: Box::new(Expr::Name { name: Identifier { name: "b".to_owned() }}),
                                                                                    op: Op::Add,
                                                                                    left: Box::new(Expr::Name { name: Identifier { name: "a".to_owned() }}),
                                                                                } }
        ];

        assert_input_with_program(input, result);
    }

    #[test]
    fn expr_stmt() {
        let input = b"12 * 25;";
        let result = vec![
            Stmt::Expr {
                expr: Expr::InfixExpr {
                    op: Op::Mult,
                    left: Box::new(Expr::LiteralExpr {value: IntegerLiteral(12)}),
                    right: Box::new(Expr::LiteralExpr {value: IntegerLiteral(25)})
                }
            }
        ];
        assert_input_with_program(input, result);

        let input = b"12 - 12 / -22 + (x - x);";
        let input2 = b"12 - (12 / (-22)) + (x - x);";
        is_the_same_inputs(input, input2);

        //let input = b"10 + 5 * -21 / -10 * (a + b)";
        //let program = vec![];
        //assert_input_with_program(input, program);

    }

    #[test]
    fn array_stmts() {
        let input = b"[1, 2, 3, a + b, a * b]";
        let program = vec![
            Stmt::Expr {
                expr: Expr::ArrayExpr {
                values: vec![
                    Expr::LiteralExpr{value: IntegerLiteral(1)},
                    Expr::LiteralExpr{value: IntegerLiteral(2)},
                    Expr::LiteralExpr{value: IntegerLiteral(3)},
                    Expr::InfixExpr {
                        op: Op::Add,
                        left: Box::new(Expr::Name {name: Identifier { name: "a".to_owned()}}),
                        right: Box::new(Expr::Name {name: Identifier { name: "b".to_owned()}})
                    },
                    Expr::InfixExpr {
                        op: Op::Mult,
                        left: Box::new(Expr::Name {name: Identifier { name: "a".to_owned()}}),
                        right: Box::new(Expr::Name {name: Identifier { name: "b".to_owned()}})
                    },
                ]
            }}
        ];
        assert_input_with_program(input, program);
    }

    #[test]
    fn if_expr() {
        let input = b"if (x > y) {
        x
        } else {if (x == y){
          0
        } else {
          y
        }}";
        let program = vec![
            Stmt::Expr {
                expr: Expr::If {
                    cond: Box::new(Expr::InfixExpr {
                        left: Box::new(Expr::Name {name: Identifier {name: "x".to_owned()}}),
                        op: Op::Gt,
                        right: Box::new(Expr::Name {name: Identifier {name: "y".to_owned()}})
                    }),
                    body: vec![
                        Stmt::Expr {
                            expr: Expr::Name {name: Identifier {name: "x".to_owned()}}
                        }
                    ],
                    or_else: Some(vec![
                        Stmt::Expr {
                            expr: Expr::If {
                                cond: Box::new(Expr::InfixExpr {
                                    left: Box::new(Expr::Name {name: Identifier {name: "x".to_owned()}}),
                                    op: Op::Eq,
                                    right: Box::new(Expr::Name {name: Identifier {name: "y".to_owned()}})
                                }),
                                body: vec![
                                    Stmt::Expr {
                                        expr: Expr::LiteralExpr{value: IntegerLiteral(0)},
                                    }
                                ],
                                or_else: Some(vec![
                                    Stmt::Expr {expr: Expr::Name { name: Identifier { name: "y".to_owned()}}}
                                ])
                            }
                        }
                    ])
                }
            }
        ];
        assert_input_with_program(input, program);
    }

}