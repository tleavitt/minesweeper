use std::cmp::{min, max};
///
/// Common grid functions
///

pub fn get<T>(grid: &Vec<Vec<T>>, i: usize, j: usize) -> &T {
    &grid[i][j]
}

pub fn get_mut<T>(grid: &mut Vec<Vec<T>>, i: usize, j: usize) -> &mut T {
    &mut grid[i][j]
}

pub fn get_num_rows<T>(grid: &Vec<Vec<T>>) -> usize {
    grid.len()
}

pub fn get_num_cols<T>(grid: &Vec<Vec<T>>) -> usize {
    grid[0].len()
}

pub fn get_row_col_str<T>(grid: &Vec<Vec<T>>) -> String {
    format!("nrow: {}, ncols: {}\n",
            get_num_rows(grid),
            get_num_cols(grid)
    )
}

/// Returns a list of all valid neighbors of the given cell
/// TODO: is allocating and returning a list for this all the time inefficient?
pub fn get_neighbors<T>(grid: &Vec<Vec<T>>, i: usize, j: usize) -> Vec<(usize, usize)> {
    let mut neighbors: Vec<(usize, usize)> = Vec::with_capacity(8);
    let mut add_neighbor = |_: &T, ni: usize, nj: usize| {
        neighbors.push((ni, nj));
    };
    for_each_neighbor(grid, i, j, &mut add_neighbor);
    neighbors
}

pub fn for_each_neighbor<T>(grid: &Vec<Vec<T>>, i: usize, j: usize, f: &mut dyn for<'r> FnMut(&'r T, usize, usize) -> ()) {
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
                f(get(grid, ni, nj), ni, nj)
            }
        }
    }
}