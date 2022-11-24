use crate::count_grid::*;
use crate::mine_map::*;
use crate::count_grid;
use crate::grid::*;

#[derive(Debug, Clone)]
pub struct SolveCell {
    mine_likelihood: f64, // The probability that this cell is a mine
}

// Internal board used by the solver
type SolveGrid = Vec<Vec<SolveCell>>;

// Container for data structures the solver uses.
pub struct SolveState {
    solve_grid: SolveGrid,
    count_grid: CountGrid,
    nmines: usize,
    mines_found: usize,
}

impl SolveState {
    pub fn init(nrows: usize, ncols: usize, nmines: usize) -> SolveState {
        SolveState {
            solve_grid: init_solve_grid(nrows, ncols, nmines),
            count_grid: init_count_grid(nrows, ncols),
            nmines,
            mines_found: 0,
        }
    }
}

pub fn init_solve_grid(nrows: usize, ncols: usize, nmines: usize) -> SolveGrid {
    let mut solve_grid: SolveGrid = Vec::with_capacity(nrows);
    let prior: f64 = nmines as f64 / (nrows * ncols) as f64;
    for _ in 0..nrows {
        solve_grid.push(vec![SolveCell {mine_likelihood: prior}; ncols]);
    }
    solve_grid
}

///
/// Terms:
/// settled cell: a cell that's either a known mine, or an marked cell where we know which of its neighbors are mines (if any)
/// boundary cell: marked cell with a non-zero mine count that's not settled (i.e. we're not sure what all of it's mines are)
/// frontier cell: unmarked, unsettled cell that borders a boundary cell (i.e. a candidate for being a mine)
///
/// Marks a cell, updating the count grid and the solve grid,
/// and returning a list of new boundary cells revealed by the mark
pub fn mark(solve_state: &mut SolveState, mark_i: usize, mark_j: usize, mine_map: &MineMap) -> Vec<(usize, usize)> {
    let marked_cells = count_grid::mark(&mut solve_state.count_grid, mark_i, mark_j, mine_map);
    let mut new_boundary_cells: Vec<(usize, usize)> = Vec::with_capacity(marked_cells.len());

    // First, set all marked cells to have zero likelihood.
    for (i, j) in &marked_cells {
        // Marked cells are not mines, so set their likelihoods to zero.
        get_mut(&mut solve_state.solve_grid, *i, *j).mine_likelihood = 0.0;
    }
    for cur in &marked_cells {
        let (i, j) = cur;
        // Check if this is a boundary cell, and if so, add it to our output.
        if get(&solve_state.count_grid, *i, *j).value > 0 {
            // Will be a boundary cell if it's not settled - i.e. it has some unsettled neighbors
            if !get_unsettled_neighbors(&solve_state.solve_grid, *i, *j).is_empty() {
                new_boundary_cells.push(*cur);
            }
        }
    }
    new_boundary_cells
}

fn get_unsettled_neighbors(solve_grid: &SolveGrid, i: usize, j: usize) -> Vec<(usize, usize)>{
    let mut unsettled_neighbors = Vec::with_capacity(8);
    for_each_neighbor(solve_grid, i, j, &mut |cell, ni, nj| {
        if cell.mine_likelihood != 0.0 && cell.mine_likelihood != 1.0 {
            unsettled_neighbors.push((i, j))
        }
    });
    unsettled_neighbors
}

pub fn to_string(solve_grid: &SolveGrid) -> String {
    let mut str = get_row_col_str(solve_grid);
    for row in solve_grid {
        for cell in row {
            str.push_str(&*format!("{:.2} ", cell.mine_likelihood));
        }
        str.push('\n');
    }
    str
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mine_map::MineMap;
    use crate::solve_grid;

    #[test]
    fn test_update_likelihoods1() {
        let mine_map: MineMap = vec![
            vec![true, false, false, false, true],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, false, false, true],
        ];
        let mut solve_state = SolveState::init(
            get_num_rows(&mine_map),
            get_num_cols(&mine_map),
            3
        );
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", solve_grid::to_string(&solve_state.solve_grid));
        let marked_cells = mark(&mut solve_state, 0, 2, &mine_map);
        println!("{}", count_grid::to_string(&solve_state.count_grid));
        println!("{}", solve_grid::to_string(&solve_state.solve_grid));

        assert_eq!(
            vec![
                vec![-1,  1,  0,  1, -1],
                vec![ 1,  1,  0,  1,  1],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  1,  1],
                vec![ 0,  0,  0,  1, -1],
            ],
            flatten_cells(&solve_state.count_grid)
        )
    }
}