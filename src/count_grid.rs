use crate::grid::*;
use crate::mine_map::{get_neighbor_mine_count, MineMap};
use std::collections::{HashSet};

// Board shown to the user, kept independently from the mine map itself.
#[derive(Debug, Clone)]
pub struct CountCell {
    pub neighbor_mine_count: i32
}
impl CountCell {
    pub fn is_marked(&self) -> bool {
        return self.neighbor_mine_count != UNKNOWN_CELL_COUNT
    }

    pub fn is_boundary_cell(&self) -> bool {
        return self.neighbor_mine_count > 0
    }
}
pub type CountGrid = Vec<Vec<CountCell>>;

const UNKNOWN_CELL_COUNT: i32 = -1;

pub fn init_count_grid(nrows: usize, ncols: usize) -> CountGrid {
    let mut count_grid: CountGrid = Vec::with_capacity(nrows);
    for _ in 0..nrows {
        count_grid.push(vec![CountCell { neighbor_mine_count: UNKNOWN_CELL_COUNT }; ncols]);
    }
    count_grid
}

/// Mark a single cell and reveal its mine count
fn mark_cell(count_grid: &mut CountGrid, i: usize, j: usize, mine_map: &MineMap) -> i32 {
    let mut cur_cell: &mut CountCell = get_mut(count_grid, i, j);
    if cur_cell.is_marked() {
        panic!("Cell {},{} has already been marked", i, j);
    }
    cur_cell.neighbor_mine_count = get_neighbor_mine_count(mine_map, i, j) as i32;
    cur_cell.neighbor_mine_count
}

/// Mark a cell and iteratively mark all adjacent empty cells.
/// Returns a list of all cells marked
pub fn mark(count_grid: &mut CountGrid, i: usize, j: usize, mine_map: &MineMap) -> Vec<(usize, usize)> {
    // If we try to mark a mine, panic. This should be caught earlier.
    if *get(mine_map, i, j) {
        panic!("marked cell {i},{j} is a mine");
    }
    // mark_iterative(count_grid, i, j, mine_map)
    mark_recursive(count_grid, i, j, mine_map)
}

fn mark_iterative(count_grid: &mut CountGrid, i: usize, j: usize, mine_map: &MineMap) -> Vec<(usize, usize)> {
    // Use a hash-set to store the mark queue to avoid double-marking
    let mut to_mark: HashSet<(usize, usize)> = HashSet::with_capacity(8);
    let mut marked: Vec<(usize, usize)> = Vec::with_capacity(8);
    to_mark.insert((i, j));
    while !to_mark.is_empty() {
        // My hacky way to "pop" the first element off a hash set -
        // this is perhaps inefficient
        let cur = *to_mark.iter().next().unwrap();
        to_mark.remove(&cur);
        let (cur_i, cur_j) = cur;
        let mine_count = mark_cell(count_grid, cur_i, cur_j, mine_map);
        marked.push(cur);
        // TODO: there's perhaps potential for optimization here (running over the neighbors
        //  twice)?
        if mine_count == 0 {
            // None of the neighbors are mines, so enqueue all the unmarked neighbors for marking.
            for cur_neigh in get_neighbors(mine_map, cur_i, cur_j) {
                let (ni, nj) = cur_neigh;
                if !get(count_grid, ni, nj).is_marked() {
                    to_mark.insert(cur_neigh); // The hash-set-ness will deduplicate for us.
                }
            }
        }
    }
    marked
}

fn mark_recursive(count_grid: &mut CountGrid, i: usize, j: usize, mine_map: &MineMap) -> Vec<(usize, usize)> {
    let mut marked: Vec<(usize, usize)> = Vec::with_capacity(8);
    mark_recursive_impl(count_grid, i, j, mine_map, &mut marked);
    marked
}

fn mark_recursive_impl(count_grid: &mut CountGrid, i: usize, j: usize, mine_map: &MineMap, marked: &mut Vec<(usize, usize)>) {
    // Step 1: mark the current cell
    let mine_count = mark_cell(count_grid, i, j, mine_map);
    // Step 2: record that we marked this cell
    marked.push((i, j));
    // Step 3: if no adjacent mines, mark all unmarked neighbors and recurse
    if mine_count == 0 {
        for (ni, nj) in get_neighbors(mine_map, i, j) {
            if !get(count_grid, ni, nj).is_marked() {
               mark_recursive_impl(count_grid, ni, nj, mine_map, marked);
            }
        }
    }
}

pub fn flatten_cells(count_grid: &CountGrid) -> Vec<Vec<i32>> {
    count_grid.iter().map(
        |row| row.iter().map(
            |cell| cell.neighbor_mine_count
        ).collect()
    ).collect()
}

pub fn from_vec(count_grid_vec: Vec<Vec<i32>>) -> CountGrid {
    count_grid_vec.iter().map(
        |row| row.iter().map(
            |count| CountCell {neighbor_mine_count: *count}
        ).collect()
    ).collect()
}

pub fn to_string(count_grid: &CountGrid) -> String {
    let mut str = get_row_col_str(count_grid);
    for row in count_grid {
        for cell in row {
            if cell.neighbor_mine_count == UNKNOWN_CELL_COUNT {
                str.push_str("- ");
            } else {
               str.push_str(&*format!("{} ", cell.neighbor_mine_count));
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
        let mut count_grid: CountGrid = init_count_grid(3, 3);
        println!("{}", to_string(&count_grid));
        mark_cell(&mut count_grid, 0, 1, &mine_map);
        println!("{}", to_string(&count_grid));
        mark_cell(&mut count_grid, 2, 2, &mine_map);
        println!("{}", to_string(&count_grid));
        mark_cell(&mut count_grid, 1, 1, &mine_map);
        println!("{}", to_string(&count_grid));

        assert_eq!(
            vec![
                vec![-1, 3, -1],
                vec![-1, 2, -1],
                vec![-1, -1, 1],
            ],
            flatten_cells(&count_grid)
        )
    }

    #[test]
    fn test_mark1() {
        let mine_map: MineMap = vec![
            vec![true, false, false],
            vec![true, true, false],
            vec![false, false, false],
        ];
        let mut count_grid: CountGrid = init_count_grid(3, 3);
        println!("{}", to_string(&count_grid));
        mark(&mut count_grid, 0, 1, &mine_map);
        println!("{}", to_string(&count_grid));
        mark(&mut count_grid, 2, 2, &mine_map);
        println!("{}", to_string(&count_grid));
        mark(&mut count_grid, 1, 1, &mine_map);
        println!("{}", to_string(&count_grid));

        assert_eq!(
            vec![
                vec![-1, 3, -1],
                vec![-1, 2, -1],
                vec![-1, -1, 1],
            ],
            flatten_cells(&count_grid)
        )
    }

    #[test]
    fn test_mark2() {
        let mine_map: MineMap = vec![
            vec![true,  false, false, false],
            vec![true,  true,  false, false],
            vec![false, false, false, false],
        ];
        let mut count_grid: CountGrid = init_count_grid(3, 4);
        println!("{}", to_string(&count_grid));
        mark(&mut count_grid, 1, 3, &mine_map);
        println!("{}", to_string(&count_grid));

        assert_eq!(
           vec![
                vec![-1, -1, 1, 0],
                vec![-1, -1, 1, 0],
                vec![-1, -1, 1, 0],
            ],
           flatten_cells(&count_grid)
        )

    }

    #[test]
    fn test_mark3() {
        let mine_map: MineMap = vec![
            vec![true,  false, false, false, true ],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, true ],
        ];
        let mut count_grid: CountGrid = init_count_grid(get_num_rows(&mine_map), get_num_cols(&mine_map));
        println!("{}", to_string(&count_grid));
        mark(&mut count_grid, 0, 2, &mine_map);
        println!("{}", to_string(&count_grid));

        assert_eq!(
            vec![
                vec![-1,  1,  0,  1, -1],
                vec![ 1,  1,  0,  1,  1],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  1,  1],
                vec![ 0,  0,  0,  1, -1],
            ],
            flatten_cells(&count_grid)
        )
    }

    #[test]
    fn test_mark4() {
        let mine_map: MineMap = vec![
            vec![true,  false, false, false, true ],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, true ],
        ];
        let mut count_grid: CountGrid = init_count_grid(get_num_rows(&mine_map), get_num_cols(&mine_map));
        println!("{}", to_string(&count_grid));
        mark(&mut count_grid, 5, 2, &mine_map);
        println!("{}", to_string(&count_grid));

        assert_eq!(
            vec![
                vec![-1,  1,  0,  1, -1],
                vec![ 1,  1,  0,  1,  1],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  1,  1],
                vec![ 0,  0,  0,  1, -1],
            ],
            flatten_cells(&count_grid)
        )
    }

    #[test]
    fn test_mark5() {
        let mine_map: MineMap = vec![
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
        ];
        let mut count_grid: CountGrid = init_count_grid(get_num_rows(&mine_map), get_num_cols(&mine_map));
        println!("{}", to_string(&count_grid));
        mark(&mut count_grid, 4, 4, &mine_map);
        println!("{}", to_string(&count_grid));

        assert_eq!(
            vec![
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
            ],
            flatten_cells(&count_grid)
        )
    }
}
