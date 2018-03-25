extern crate kusuo_lib;
#[macro_use]
extern crate clap;

use clap::{Arg, App};
use std::fs::File;
use std::path::Path;
use kusuo_lib::lexer::Lexer;
use std::io::Read;

fn main() {
    let version = "0.0.1";
    let matches = App::new("kusuo compiler")
        .version(version)
        .author("ariwaranosai <nkssai@outlook.com>")
        .arg(Arg::with_name("lexer")
            .short("l")
            .long("lexer")
            .help("compile only with lexer"))
        .arg(Arg::with_name("input_file")
            .short("i")
            .value_name("input_file")
            .help("sets the input file to compile")
            .required(true))
        .get_matches();

    let file_path = matches.value_of("input_file").unwrap();

    if matches.is_present("lexer") {
        run_lexer(file_path);
    }
}

fn run_lexer(file_path: &str){
    let path = Path::new(file_path);
    let file_name = path.file_name().unwrap();
    let mut f = File::open(file_path).expect("file not found!");
    let mut content = String::new();
    f.read_to_string(&mut content).expect("read file error!");

    let tokens = Lexer::lex_tokens(content.as_ref());

    println!("{}", tokens.to_result().unwrap().iter()
        .map(|x| {format!("{:?}", x)})
        .collect::<Vec<_>>().join("\n"))
}