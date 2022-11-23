use rand::prelude::*;
use crate::grid::{get_num_cols, get_num_rows};

/// Ground-truth representation of a game (i.e. where the mines are)
type MineMap = Vec<Vec<bool>>;


///
/// Generates a new nrows x ncols mine map with nmines mines
/// distributed across the map uniformly at random.
///
pub fn generate_new_mine_map(nrows: usize, ncols: usize, nmines: usize) -> MineMap {
    let ncells = nrows * ncols;
    if nmines > ncells {
        panic!("too many mines - mine map of size {} x {} can have at most {} mines, requested {}", nrows, ncols, ncells, nmines);
    }
    let mine_cells = {
        let mut rng = rand::thread_rng();
        // Algorithm for picking nmines cells at random from the ncells remaining cells:
        // Starting from i = 0, pick index i with probability n_left_to_pick / n_left_to_pick_from
        // So we'd choose 0 with probability nmines / ncells, then 1 with probability
        // nmines - 1 / ncells -1 if we picked 0, else nmines / ncells - 1, etc.
        let mut mine_cells: Vec<usize> = Vec::with_capacity(nmines);
        for cell_idx in 0..ncells {
            let n_left_to_pick = (nmines - mine_cells.len()) as f64;
            let n_left_to_pick_from = (ncells - cell_idx) as f64;
            // Pick cell_idx as a mine with probability n_left_to_pick / n_left_to_pick_from
            if rng.gen_bool(n_left_to_pick / n_left_to_pick_from) {
                mine_cells.push(cell_idx);
            }
        }
        mine_cells
    };

    // Convert an index from 0...nrows*ncols to the corresponding (row, col) tuple.
    // We use row-major order
    let cell_idx_to_row_col = |cell_idx: usize| -> (usize, usize) {
        (cell_idx / ncols, cell_idx % ncols)
    };
    let mut next_cell_idx = 0;
    let mut get_next_mine_coords = || -> (usize, usize) {
        if next_cell_idx >= mine_cells.len() {
            // Return a sentinel value that will never be reached
            return (nrows, ncols);
        }
        let coords = cell_idx_to_row_col(mine_cells[next_cell_idx]);
        next_cell_idx += 1;
        coords
    };
    let mut next_mine_coords = get_next_mine_coords();

    let mut mine_map: MineMap = Vec::with_capacity(nrows);

    // We could interleave this with the row-picking algorithm for very slight performance
    // improvements, but I think it's clearer written this way.
    for row in 0..nrows {
        let mut cur_row: Vec<bool> = Vec::with_capacity(ncols);
        for col in 0..ncols {
            if (row, col) == next_mine_coords {
                cur_row.push(true);
                next_mine_coords = get_next_mine_coords()
            } else {
                cur_row.push(false);
            }
        }
        mine_map.push(cur_row);
    }
    mine_map
}

pub fn to_string(mine_map: &MineMap) -> String {
    let mut str = String::from(
        format!("nrow: {}, ncols: {}\n",
                get_num_rows(&mine_map),
                get_num_cols(&mine_map)
        ));
    for row in mine_map {
        for cell in row {
            str.push_str(if *cell { "x " } else { "- " });
        }
        str.push('\n');
    }
    str
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_mine_map(mine_map: &MineMap, nrows: usize, ncols: usize, nmines: usize) {
        let mut nmines_seen: usize = 0;
        assert_eq!(nrows, mine_map.len());
        for row in mine_map {
            assert_eq!(ncols, row.len());
            for cell in row {
                if *cell {
                    nmines_seen += 1;
                }
            }
        }
        assert_eq!(nmines, nmines_seen);
    }

    #[test]
    fn test_generate_small_mine_map() {
        for _ in 0..10 {
            let mine_map = generate_new_mine_map(3, 3, 3);
            println!("{}", to_string(&mine_map));
            validate_mine_map(&mine_map, 3, 3, 3);
        }
    }

    #[test]
    fn test_generate_large_mine_map() {
        for _ in 0..10 {
            let mine_map = generate_new_mine_map(40, 40, 40);
            println!("{}", to_string(&mine_map));
            validate_mine_map(&mine_map, 40, 40, 40);
        }
    }

    #[test]
    fn test_generate_empty_mine_map() {
        let mine_map = generate_new_mine_map(3, 3, 0);
        println!("{}", to_string(&mine_map));
        validate_mine_map(&mine_map, 3, 3, 0)
    }

    #[test]
    fn test_generate_full_mine_map() {
        let mine_map = generate_new_mine_map(4, 3, 12);
        println!("{}", to_string(&mine_map));
        validate_mine_map(&mine_map, 4, 3, 12);
    }

    #[should_panic]
    #[test]
    fn test_generate_overfull_mine_map() {
        let mine_map = generate_new_mine_map(4, 3, 13);
        println!("{}", to_string(&mine_map));
        validate_mine_map(&mine_map, 4, 3, 12);
    }
}