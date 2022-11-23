use std::cmp::{min, max};
///
/// Common grid functions
///

pub fn get<T>(grid: &Vec<Vec<T>>, i: usize, j: usize) -> &T {
    &grid[i][j]
}

pub fn set<T>(grid: &mut Vec<Vec<T>>, i: usize, j: usize, value: T) {
    &grid[i][j] = value;
}

pub fn get_num_rows<T>(grid: &Vec<Vec<T>>) -> usize {
    grid.len()
}

pub fn get_num_cols<T>(grid: &Vec<Vec<T>>) -> usize {
    grid[0].len()
}

/// Returns a list of all valid neighbors of the given cell
pub fn get_neighbors<T>(grid: &Vec<Vec<T>>, i: usize, j: usize) -> Vec<(usize, usize)> {
    let mut neighbors: Vec<(usize, usize)> = Vec::with_capacity(8);
    let nrows = get_num_rows(grid);
    let ncols = get_num_cols(grid);
    // Note: all values are inclusive
    let min_i = if i <= 0 { 0 } else { i-1 };
    let max_i = if i >= nrows-1 { nrows-1 } else { i+1 };
    let min_j = if j <= 0 { 0 } else { j-1 };
    let max_j = if j >= ncols-1 { ncols-1 } else { j+1 };

    for ni in min_i..max_i+1 {
        for nj in min_j..max_j+1 {
            if !(ni == i && nj == j) {
                neighbors.push((ni, nj));
            }
        }
    }

    neighbors
}
