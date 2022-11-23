use crate::grid::{get, set};
use crate::mine_map::{get_neighbor_mine_count, MineMap};

// Board shown to the user, kept independently from the mine map itself.
type GameCell = i32;
type GameBoard = Vec<Vec<GameCell>>;

const UNKNOWN_CELL: GameCell = -1;

fn init_game_board(nrows: usize, ncols: usize) -> GameBoard {
    let mut game_board: GameBoard = Vec::with_capacity(nrows);
    for _ in 0..ncols {
        game_board.push(vec![UNKNOWN_CELL; ncols]);
    }
    game_board
}

fn mark(game_board: &mut GameBoard, i: usize, j: usize, mine_map: &MineMap) {
    if *get(game_board, i, j) != UNKNOWN_CELL {
        panic!("Cell {},{} has already been marked", i, j);
    }
    set(game_board, i, j, get_neighbor_mine_count(mine_map, i, j) as GameCell);
}

