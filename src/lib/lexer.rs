use std::ops::Range;
use std::ops::RangeTo;
use std::ops::RangeFrom;
use std::ops::RangeFull;
use std::iter::Enumerate;
use nom::*;
use std::str;
use std::str::FromStr;
use std::str::Utf8Error;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Comment(String),
    Float(f64),
    Integer(i64),
    Char(char),
    String(String),
    Bool(bool),
    Identifier(String),
    Symbol(Symbol),
    Keyword(Keyword),
    EOF,
    Illegal // for error
}

#[derive(Debug, Clone, PartialEq)]
pub enum Keyword {
    While,
    If,
    Else,
    Break,
    True,
    False,
    Def,
    Continue,
    Return
}

#[derive(Debug, Clone, PartialEq)]
pub enum Symbol {
    LParenthesis, // (
    RParenthesis, // )
    LBrace, // {
    RBrace, // }
    LBracket, // [
    RBracket, // ]
    LineEnd, // \n
    SemiColon, // ;
    Assign, // =
    Equal, // ==
    NotEqual, // !=
    GT, // >
    GE, // >=
    LT, // <
    LE, // <=
    Plus, // +
    Minus, // -
    Mult, // *
    Div, // /
    Mod, // %
    Not, // !
    Comma // ,
}

#[derive(Debug, Clone, PartialEq, Copy)]
pub struct Tokens<'a> {
    pub toks: &'a [Token],
    pub start: usize,
    pub end: usize
}

impl<'a> Tokens<'a> {
    pub fn new(vec: &'a Vec<Token>) -> Self {
        Tokens {
            toks: vec.as_slice(),
            start: 0,
            end: vec.len(),
        }
    }
}

impl<'a> InputLength for Tokens<'a> {
    #[inline(always)]
    fn input_len(&self) -> usize {
        self.toks.len()
    }
}

impl InputLength for Token {
    #[inline(always)]
    fn input_len(&self) -> usize { 1 }
}

impl<'a> Slice<Range<usize>> for Tokens<'a> {
    #[inline]
    fn slice(&self, range: Range<usize>) -> Self {
        Tokens {
            toks: self.toks.slice(range.clone()),
            start: self.start + range.start,
            end: self.start + range.end
        }
    }
}

impl<'a> Slice<RangeTo<usize>> for Tokens<'a> {
    #[inline]
    fn slice(&self, range: RangeTo<usize>) -> Self {
        self.slice(0..range.end)
    }
}

impl<'a> Slice<RangeFrom<usize>> for Tokens<'a> {
    #[inline]
    fn slice(&self, range: RangeFrom<usize>) -> Self {
        self.slice(range.start..self.end - self.start)
    }
}

impl<'a> Slice<RangeFull> for Tokens<'a> {
    #[inline]
    fn slice(&self, _: RangeFull) -> Self {
        Tokens {
            toks: self.toks,
            start: self.start,
            end: self.end,
        }
    }
}

impl<'a> InputIter for Tokens<'a> {
    type Item = &'a Token;
    type RawItem = Token;
    type Iter = Enumerate<::std::slice::Iter<'a, Token>>;
    type IterElem = ::std::slice::Iter<'a, Token>;

    #[inline]
    fn iter_indices(&self) -> Enumerate<::std::slice::Iter<'a, Token>> {
        self.toks.iter().enumerate()
    }
    #[inline]
    fn iter_elements(&self) -> ::std::slice::Iter<'a, Token> {
        self.toks.iter()
    }
    #[inline]
    fn position<P>(&self, predicate: P) -> Option<usize>
        where
            P: Fn(Self::RawItem) -> bool,
    {
        self.toks.iter().position(|b| predicate(b.clone()))
    }
    #[inline]
    fn slice_index(&self, count: usize) -> Option<usize> {
        if self.toks.len() >= count {
            Some(count)
        } else {
            None
        }
    }
}


// todo generate operator lexer by macros
// operator
named!(left_parenthesis<&[u8], Token>,
 do_parse!(tag!("(") >> (Token::Symbol(Symbol::LParenthesis)))
);

named!(right_parenthesis<&[u8], Token>,
 do_parse!(tag!(")") >> (Token::Symbol(Symbol::RParenthesis)))
);

named!(left_bracket<&[u8], Token>,
 do_parse!(tag!("[") >> (Token::Symbol(Symbol::LBracket)))
);

named!(right_bracket<&[u8], Token>,
 do_parse!(tag!("]") >> (Token::Symbol(Symbol::RBracket)))
);

named!(left_brace<&[u8], Token>,
 do_parse!(tag!("{") >> (Token::Symbol(Symbol::LBrace)))
);

named!(right_brace<&[u8], Token>,
 do_parse!(tag!("}") >> (Token::Symbol(Symbol::RBrace)))
);


named!(line_end<&[u8], Token>,
 do_parse!(tag!("\n") >> (Token::Symbol(Symbol::LineEnd)))
);

named!(semi_colon<&[u8], Token>,
 do_parse!(tag!(";") >> (Token::Symbol(Symbol::SemiColon)))
);

named!(assign_op<&[u8], Token>,
 do_parse!(tag!("=") >> (Token::Symbol(Symbol::Assign)))
);

named!(equal_op<&[u8], Token>,
  do_parse!(tag!("==") >> (Token::Symbol(Symbol::Equal)))
);

named!(not_equal_op<&[u8], Token>,
  do_parse!(tag!("!=") >> (Token::Symbol(Symbol::NotEqual)))
);

named!(greater_than<&[u8], Token>,
 do_parse!(tag!(">") >> (Token::Symbol(Symbol::GT)))
);

named!(greater_eq<&[u8], Token>,
 do_parse!(tag!(">=") >> (Token::Symbol(Symbol::GE)))
);

named!(lesser_than<&[u8], Token>,
 do_parse!(tag!("<") >> (Token::Symbol(Symbol::LT)))
);

named!(lesser_eq<&[u8], Token>,
 do_parse!(tag!("<=") >> (Token::Symbol(Symbol::LE)))
);

named!(plus_op<&[u8], Token>,
 do_parse!(tag!("+") >> (Token::Symbol(Symbol::Plus)))
);

named!(minus_op<&[u8], Token>,
 do_parse!(tag!("-") >> (Token::Symbol(Symbol::Minus)))
);

named!(mult_op<&[u8], Token>,
 do_parse!(tag!("*") >> (Token::Symbol(Symbol::Mult)))
);

named!(div_op<&[u8], Token>,
 do_parse!(tag!("/") >> (Token::Symbol(Symbol::Div)))
);

named!(mod_op<&[u8], Token>,
 do_parse!(tag!("%") >> (Token::Symbol(Symbol::Mod)))
);

named!(not_op<&[u8], Token>,
 do_parse!(tag!("!") >> (Token::Symbol(Symbol::Not)))
);

named!(comma<&[u8], Token>,
 do_parse!(tag!(",") >> (Token::Symbol(Symbol::Comma)))
);
// all operators

named!(lex_operator<&[u8], Token>, alt!(
          equal_op|
      not_equal_op|
        greater_eq|
         lesser_eq|
      greater_than|
       lesser_than|
           plus_op|
          minus_op|
          mult_op |
            div_op|
            mod_op|
            not_op|
          line_end|
         assign_op
));

// punctuation
named!(lex_punctuation<&[u8], Token>, alt!(
 left_parenthesis |
 right_parenthesis|
        left_brace|
       right_brace|
        semi_colon
));


// keywords and identifier
fn parse_keywords_ident(c: &str, rest: Option<&str>) -> Token {
    let mut s = c.to_owned();
    s.push_str(rest.unwrap_or(""));
    match s.as_ref() {
        "if" => Token::Keyword(Keyword::If),
        "else" => Token::Keyword(Keyword::Else),
        "while" => Token::Keyword(Keyword::While),
        "def" => Token::Keyword(Keyword::Def),
        "true" => Token::Bool(true),
        "false" => Token::Bool(false),
        "return" => Token::Keyword(Keyword::Return),
        _ => Token::Identifier(s)
    }
}

macro_rules! check(
  ($input:expr, $submac:ident!( $($args:tt)* )) => (
    {
      let mut failed = false;
      for &idx in $input {
        if !$submac!(idx, $($args)*) {
            failed = true;
            break;
        }
      }
      if failed {
        IResult::Error(ErrorKind::Custom(20))
      } else {
        IResult::Done(&b""[..], $input)
      }
    }
  );
  ($input:expr, $f:expr) => (
    check!($input, call!($f));
  );
);

named!(take_1_char, flat_map!(take!(1), check!(is_alphabetic)));

named!(lex_reserved_ident<&[u8], Token>,
    do_parse!(
        c: map_res!(call!(take_1_char), str::from_utf8) >>
        rest: opt!(complete!(map_res!(alphanumeric, str::from_utf8))) >>
        (parse_keywords_ident(c, rest))
    )
);

// integer
named!(lex_integer<&[u8], Token>,
    do_parse!(
      number: alt_complete!(
        map_opt!(preceded!(tag!("0x"), hex_digit), |v| vu8_to_token(v, 16)) |
        map_opt!(preceded!(tag!("0o"), oct_digit), |v| vu8_to_token(v, 8)) |
        map_opt!(preceded!(tag!("0b"), take_while!(is_one_or_zero)), |v| vu8_to_token(v, 2)) |
        map_opt!(digit, |v| vu8_to_token(v, 10))
      ) >> (u32_to_token(number))
));

#[inline]
fn vu8_to_token(n: &[u8], radix: u32) -> Option<i64> {
    let s = str::from_utf8(n).unwrap();
    i64::from_str_radix(s, radix).ok()
}

#[inline]
fn u32_to_token(number: i64) -> Token {
    Token::Integer(number)
}

#[inline]
fn is_one_or_zero(c: u8) -> bool {
    c == b'0' || c == b'1'
}

// float
named!(parse_float_exp, recognize!(do_parse!(
               alt!(tag!("e") | tag!("E"))
            >> opt!(alt!(tag!("+") | tag!("-")))
            >> digit
            >> ())));

named!(lex_float<&[u8], Token>,
       do_parse!(
              float: map_res!( recognize!( do_parse!(
                         alt!(
                          delimited!(digit, tag!("."), opt!(complete!(digit))) |
                          delimited!(opt!(digit), tag!("."), digit) |
                          digit)
                      >> opt!(complete!(parse_float_exp))
                      >> ())),
                  str::from_utf8)
           >> (parse_float_token(float)))
);

#[inline]
fn parse_float_token(float: &str) -> Token {
    Token::Float(f64::from_str(float).unwrap())
}

// Strings
fn pis(input: &[u8]) -> IResult<&[u8], Vec<u8>> {
    let (i1, c1) = try_parse!(input, take!(1));
    match c1 {
        b"\"" => IResult::Done(input, vec![]),
        b"\\" => {
            let (i2, c2) = try_parse!(i1, take!(1));
            pis(i2).map(|done| concat_slice_vec(c2, done))
        }
        c => pis(i1).map(|done| concat_slice_vec(c, done)),
    }
}

fn concat_slice_vec(c: &[u8], done: Vec<u8>) -> Vec<u8> {
    let mut new_vec = c.to_vec();
    new_vec.extend(&done);
    new_vec
}

fn convert_vec_utf8(v: Vec<u8>) -> Result<String, Utf8Error> {
    let slice = v.as_slice();
    str::from_utf8(slice).map(|s| s.to_owned())
}

named!(string<String>,
  delimited!(
    tag!("\""),
    map_res!(pis, convert_vec_utf8),
    tag!("\"")
  )
);

named!(lex_string<&[u8], Token>,
    do_parse!(
        s: string >>
        (Token::String(s))
    )
);

// char
named!(lex_char<&[u8], Token>,
    delimited!(
        char!('\''),
        map!(pic, |x| Token::Char(x)),
        char!('\'')
    )
);

fn pic(input: &[u8]) -> IResult<&[u8], char> {
    let (i1, c1) = try_parse!(input, take!(1));
    match c1 {
        b"\\" => {
            let (i2, c2) = try_parse!(i1, take!(1));
            match c2 {
                b"a" => IResult::Done(i2, '\x07'),
                b"b" => IResult::Done(i2, '\x08'),
                b"f" => IResult::Done(i2, '\x0c'),
                b"n" => IResult::Done(i2, '\n'),
                b"r" => IResult::Done(i2, '\r'),
                b"t" => IResult::Done(i2, '\t'),
                b"v" => IResult::Done(i2, '\x0b'),
                b"\\" => IResult::Done(i2, '\\'),
                _ => IResult::Error(ErrorKind::Char)
            }
        }
        b"\'" => IResult::Error(ErrorKind::Char),
        c => IResult::Done(i1, char::from(c[0]))
    }
}


// Illegal tokens
named!(lex_illegal<&[u8], Token>,
    do_parse!(take!(1) >> (Token::Illegal))
);

// comment
named!(lex_line_comment<&[u8], Token>,
    preceded!(
        tag!("//"),
        map_res!(take_until!("\n"), |x| {
            str::from_utf8(x).map(|y| Token::Comment(y.to_owned()))
        })
    )
);

named!(lex_multiline_comment<&[u8], Token>,
    delimited!(
        tag!("/*"),
        map_res!(take_until!("*/"), |x| {
            str::from_utf8(x).map(|y| Token::Comment(y.to_owned()))
        }),
        tag!("*/")
    )
);

named!(lex_comment<&[u8], Token>,alt_complete!(
    lex_line_comment | lex_multiline_comment
));

// all

named!(lex_token<&[u8], Token>, alt_complete!(
    lex_comment |
    lex_integer |
    lex_punctuation |
    lex_string |
    lex_reserved_ident |
    lex_operator |
    lex_char |
    lex_illegal
));

pub struct Lexer;

impl Lexer {
    pub fn lex_tokens(bytes: &[u8]) -> IResult<&[u8], Vec<Token>> {
        lex_tokens(bytes).map(|result| [&result[..], &vec![Token::EOF][..]].concat())
    }
}

named!(pub space, eat_separator!(&b" \t"[..]));

macro_rules! wsl (
    ($i:expr, $($args:tt)*) => ({
      sep!($i, space, $($args)*)
    })
);

named!(lex_tokens<&[u8], Vec<Token>>, wsl!(many0!(lex_token)));

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_test() {
        assert_eq!(equal_op(b"==").to_result(), Ok(Token::Symbol(Symbol::Equal)));
    }

    #[test]
    fn number_test() {
        assert_eq!(lex_integer(b"123").to_result(), Ok(Token::Integer(123)));
        assert_eq!(lex_integer(b"0x123").to_result(), Ok(Token::Integer(0x123)));
        assert_eq!(lex_integer(b"0o100").to_result(), Ok(Token::Integer(0o100)));
        assert_eq!(lex_integer(b"0b100").to_result(), Ok(Token::Integer(0b100)));
        assert_eq!(lex_float(b"1.32").to_result(), Ok(Token::Float(1.32)));
        assert_eq!(lex_float(b"1.32e4").to_result(), Ok(Token::Float(1.32e4)));
    }

    #[test]
    fn string_test() {
        assert_eq!(lex_string(b"\"hello \\\n world\"").to_result(),
                   Ok(Token::String("hello \n world".to_string())));
    }

    #[test]
    fn char_test() {
        assert_eq!(lex_char(b"'a'").to_result(), Ok(Token::Char('a')));
        assert_eq!(lex_char(b"'\\\\'").to_result(), Ok(Token::Char('\\')));
        assert_eq!(lex_char(b"''").to_result(), Err(ErrorKind::Char));
    }


    #[test]
    fn comment_test() {
        assert_eq!(lex_comment(b"//line comment\ni = 2").to_result(),
                   Ok(Token::Comment("line comment".to_string())));
        assert_eq!(lex_comment(b"/* one\ntwo\n three\n */").to_result(),
                   Ok(Token::Comment(" one\ntwo\n three\n ".to_string())));
    }

    #[test]
    fn lex_reserved_ident() {
        let s = b"
        str = \"123456\" // comment
        /* hello
        world!*/
        c = '\\n'
        even = -0
        odd = 0
        i = 1
        while i < 10 {
            if i % 2 == 0 {
                even = even + i;
            } else {
                odd = odd + 1;
            }
            i = i + 1
        }
        (even + odd) * 12 / 12 - 4
        ";

        let tokens = Lexer::lex_tokens(s).to_result().unwrap();
        let result = vec![
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("str".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::String("123456".to_string()),
            Token::Comment(" comment".to_string()),
            Token::Symbol(Symbol::LineEnd),
            Token::Comment(" hello\n        world!".to_string()),
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("c".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::Char('\n'),
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("even".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::Symbol(Symbol::Minus),
            Token::Integer(0),
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("odd".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::Integer(0),
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("i".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::Integer(1),
            Token::Symbol(Symbol::LineEnd),
            Token::Keyword(Keyword::While),
            Token::Identifier("i".to_string()),
            Token::Symbol(Symbol::LT),
            Token::Integer(10),
            Token::Symbol(Symbol::LBrace),
            Token::Symbol(Symbol::LineEnd),
            Token::Keyword(Keyword::If),
            Token::Identifier("i".to_string()),
            Token::Symbol(Symbol::Mod),
            Token::Integer(2),
            Token::Symbol(Symbol::Equal),
            Token::Integer(0),
            Token::Symbol(Symbol::LBrace),
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("even".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::Identifier("even".to_string()),
            Token::Symbol(Symbol::Plus),
            Token::Identifier("i".to_string()),
            Token::Symbol(Symbol::SemiColon),
            Token::Symbol(Symbol::LineEnd),
            Token::Symbol(Symbol::RBrace),
            Token::Keyword(Keyword::Else),
            Token::Symbol(Symbol::LBrace),
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("odd".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::Identifier("odd".to_string()),
            Token::Symbol(Symbol::Plus),
            Token::Integer(1),
            Token::Symbol(Symbol::SemiColon),
            Token::Symbol(Symbol::LineEnd),
            Token::Symbol(Symbol::RBrace),
            Token::Symbol(Symbol::LineEnd),
            Token::Identifier("i".to_string()),
            Token::Symbol(Symbol::Assign),
            Token::Identifier("i".to_string()),
            Token::Symbol(Symbol::Plus),
            Token::Integer(1),
            Token::Symbol(Symbol::LineEnd),
            Token::Symbol(Symbol::RBrace),
            Token::Symbol(Symbol::LineEnd),
            Token::Symbol(Symbol::LParenthesis),
            Token::Identifier("even".to_string()),
            Token::Symbol(Symbol::Plus),
            Token::Identifier("odd".to_string()),
            Token::Symbol(Symbol::RParenthesis),
            Token::Symbol(Symbol::Mult),
            Token::Integer(12),
            Token::Symbol(Symbol::Div),
            Token::Integer(12),
            Token::Symbol(Symbol::Minus),
            Token::Integer(4),
            Token::Symbol(Symbol::LineEnd),
            Token::EOF
        ];

        assert_eq!(tokens, result)

    }
}