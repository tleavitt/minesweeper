use minesweeper_solver::{count_grid, grid, solve_grid};
use minesweeper_solver::mine_map::{generate_new_mine_map, to_string};
use minesweeper_solver::solve_grid::SolveState;

use std::io;
use std::num::ParseIntError;

fn main() {
    // solve_grid::test_update_likelihoods3();
    solve_grid::test_update_likelihoods1();
    // minesweeper_repl();
}

fn minesweeper_repl() {
    let nrows = 26;
    let ncols = 19;
    let nmines = 99;
    let mine_map = generate_new_mine_map(nrows, ncols, nmines);
    // let mine_map = vec![
    //     vec![false, false, true , false, false],
    //     vec![false, false, false, false, false],
    //     vec![true , false, false, false, false],
    //     vec![true , true , true , false, false],
    //     vec![false, false, false, false, false],
    // ];
    let mut solve_state = SolveState::init(nrows, ncols, nmines);
    println!("Let's play minesweeper");
    print(&solve_state);

    loop {
        let mut mark_str = String::new();
        println!("Enter a move as: row,col");
        io::stdin()
            .read_line(&mut mark_str)
            .expect("failed to read from stdin");

        let maybe_mark = parse_mark(&mark_str);
        let (mi, mj): (usize, usize) = match maybe_mark {
            None => {
                println!("Could not parse input as coordinates row,col: {}", mark_str);
                continue;
            },
            Some(m) => {
                println!("Marking {:?}", m);
                m
            },
        };

        // Check if we hit a mine. If so, you lose!
        if *grid::get(&mine_map, mi, mj) {
            println!("Hit a mine! at {mi},{mj}: {}", to_string(&mine_map));
            print(&solve_state);
            break
        }

        let marked_cells = count_grid::mark(&mut solve_state.count_grid, mi, mj, &mine_map);
        solve_grid::update_after_mark(&mut solve_state, &marked_cells);
        print(&solve_state);
        let recommended_move = solve_grid::choose_next_mark(&solve_state);
        match recommended_move {
            None => {
                println!("No more moves available - you win!");
                break
            },
            Some(m) => println!("Computer recommends: {:?}, mine probability: {:.2}", m,
                                grid::get(&solve_state.solve_grid, m.0, m.1).mine_likelihood)
        };
    }
}

fn parse_mark(mark_str: &str) -> Option<(usize, usize)> {
    let maybe_mark: Vec<Result<usize, ParseIntError>> = mark_str.splitn(2, ",")
        .map(|coord_str| coord_str.trim().parse::<usize>()
    ).collect();

    if maybe_mark.len() != 2 {
        return None
    }
    let row = maybe_mark[0].as_ref().ok()?;
    let col = maybe_mark[1].as_ref().ok()?;
    Some((*row, *col))
}

fn print(solve_state: &SolveState) {
    println!("grid: {}", count_grid::to_string(&solve_state.count_grid));
    // println!("mine likelihoods: {}", solve_grid::to_string(&solve_state.solve_grid));
    println!("frontier: {:?}", solve_state.frontier);
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_move() {
        assert_eq!(Some((1, 1)), parse_mark("1,1"));
    }
}