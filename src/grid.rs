///
/// Common grid functions
///

pub fn get<T>(grid: &Vec<Vec<T>>, i: usize, j: usize) -> &T {
    &grid[i][j]
}

pub fn get_num_rows<T>(grid: &Vec<Vec<T>>) -> usize {
    grid.len()
}

pub fn get_num_cols<T>(grid: &Vec<Vec<T>>) -> usize {
    grid[0].len()
}
