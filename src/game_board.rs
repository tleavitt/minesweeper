use crate::grid::*;
use crate::mine_map::{get_neighbor_mine_count, MineMap};

// Board shown to the user, kept independently from the mine map itself.
#[derive(Debug, Clone)]
pub struct GameCell {
    value: i32
}
pub type GameBoard = Vec<Vec<GameCell>>;


const UNKNOWN_GAME_CELL: i32 = -1;

pub fn init_game_board(nrows: usize, ncols: usize) -> GameBoard {
    let mut game_board: GameBoard = Vec::with_capacity(nrows);
    for _ in 0..ncols {
        game_board.push(vec![GameCell {value: UNKNOWN_GAME_CELL}; ncols]);
    }
    game_board
}

pub fn mark(game_board: &mut GameBoard, i: usize, j: usize, mine_map: &MineMap) {
    let mut cur_cell: &mut GameCell = get_mut(game_board, i, j);
    if cur_cell.value != UNKNOWN_GAME_CELL {
        panic!("Cell {},{} has already been marked", i, j);
    }
    cur_cell.value = get_neighbor_mine_count(mine_map, i, j) as i32;
}

pub fn flatten_cells(game_board: &GameBoard) -> Vec<Vec<i32>> {
    game_board.iter().map(
        |row| row.iter().map(
            |cell| cell.value
        ).collect()
    ).collect()
}

pub fn to_string(game_board: &GameBoard) -> String {
    let mut str = get_row_col_str(game_board);
    for row in game_board {
        for cell in row {
            if cell.value == UNKNOWN_GAME_CELL {
                str.push_str("- ");
            } else {
               str.push_str(&*format!("{} ", cell.value));
            };
        }
        str.push('\n');
    }
    str
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_neighbor_mine_count() {
        let mine_map: MineMap = vec![
            vec![true, false, false],
            vec![true, true, false],
            vec![false, false, false],
        ];
        let mut game_board: GameBoard = init_game_board(3, 3);
        println!("{}", to_string(&game_board));
        mark(&mut game_board, 0, 1, &mine_map);
        println!("{}", to_string(&game_board));
        mark(&mut game_board, 2, 2, &mine_map);
        println!("{}", to_string(&game_board));
        mark(&mut game_board, 1, 1, &mine_map);
        println!("{}", to_string(&game_board));

        assert_eq!(
            vec![
                vec![-1, 3, -1],
                vec![-1, 2, -1],
                vec![-1, -1, 1],
            ],
            flatten_cells(&game_board)
        )
    }

}
