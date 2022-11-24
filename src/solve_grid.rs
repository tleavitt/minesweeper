use std::collections::HashSet;
use crate::count_grid::*;
use crate::mine_map::*;
use crate::count_grid;
use crate::grid::*;

#[derive(Debug, Clone)]
pub struct SolveCell {
    mine_likelihood: f64,  // The probability that this cell is a mine
    leader: (i32, i32) // The cell who's determining the likelihood of this one - must be a neighbor
}

impl SolveCell {
    pub fn is_unknown(&self) -> bool {
        self.mine_likelihood != 0.0 && self.mine_likelihood != 1.0
    }
}


// Internal board used by the solver
type SolveGrid = Vec<Vec<SolveCell>>;

// Container for data structures the solver uses.
pub struct SolveState {
    solve_grid: SolveGrid,
    count_grid: CountGrid,
    nmines: usize,
    mines_found: usize,
    frontier: HashSet<(usize, usize)>
}

///
/// Terms:
/// unknown cell: a cell with a non-zero and non-one mine probability
/// settled cell: a cell that's either a known mine, or an marked cell where none of it's neighbors are unknown cells
/// The goal is to convert all cells to settled cells.
/// boundary cell: marked cell with a non-zero mine count that's not settled (i.e. we're not sure what all of it's mines are)
/// frontier cell: unknown cell that borders a boundary cell (i.e. a candidate for being a mine)
///
impl SolveState {
    pub fn init(nrows: usize, ncols: usize, nmines: usize) -> SolveState {
        SolveState {
            solve_grid: init_solve_grid(nrows, ncols, nmines),
            count_grid: init_count_grid(nrows, ncols),
            nmines,
            mines_found: 0,
            frontier: HashSet::with_capacity(16)
        }
    }

    pub fn is_boundary_cell(&self, i: usize, j: usize) -> bool {
        if get(&self.count_grid, i, j).neighbor_mine_count <= 0 {
            false
        } else {
            let (_, n_unknown_neighbors) = get_num_mine_and_unknown_neighbors(&self.solve_grid, i, j);
            n_unknown_neighbors > 0
        }
    }
}

pub fn init_solve_grid(nrows: usize, ncols: usize, nmines: usize) -> SolveGrid {
    let mut solve_grid: SolveGrid = Vec::with_capacity(nrows);
    let prior: f64 = nmines as f64 / (nrows * ncols) as f64;
    for _ in 0..nrows {
        solve_grid.push(vec![SolveCell {mine_likelihood: prior, leader: (-1, -1)}; ncols]);
    }
    solve_grid
}


///
/// Marks a cell, updating the count grid, the solve grid, and the frontier,
/// and returning a list of new boundary cells revealed by the mark, and the revealed boundary cell
/// with the highest count.
///
pub fn mark(solve_state: &mut SolveState, mark_i: usize, mark_j: usize, mine_map: &MineMap) -> ((i32, i32), Vec<(usize, usize)>) {
    let marked_cells = count_grid::mark(&mut solve_state.count_grid, mark_i, mark_j, mine_map);
    let mut new_boundary_cells: Vec<(usize, usize)> = Vec::with_capacity(marked_cells.len());

    // First, set all marked cells to have zero likelihood and remove them from the frontier if present.
    for cur in &marked_cells {
        let (i_, j_) = cur;
        // Marked cells are not mines, so set their likelihoods to zero.
        let mut cell = get_mut(&mut solve_state.solve_grid, *i_, *j_);
        cell.mine_likelihood = 0.0;
        cell.leader = (-1, -1); // No leader for marked cells.
        // Remove the cell from the frontier if it was present.
        solve_state.frontier.remove(cur);
    }

    let mut max_count = -1;
    let mut max_count_cell: (i32, i32) = (-1, -1);
    for cur in &marked_cells {
        let (i_, j_) = cur;
        let i = *i_; let j = *j_;
        // Check if this is a boundary cell, and if so, add it to our output.
        // Will be a boundary cell if it's got a nonzero mine count and it's not settled - i.e. it has some unsettled neighbors
        let count_cell = get(&solve_state.count_grid, i, j);
        // Sanity check
        if !count_cell.is_marked() {
            panic!("Unmarked cell returned by count_grid::mark {:?}", count_cell);
        }
        let count = count_cell.neighbor_mine_count;
        if count > 0 {
            let (_, n_unknown_neighbors) = get_num_mine_and_unknown_neighbors(&solve_state.solve_grid, i, j);
            if n_unknown_neighbors > 0 {
                if count > max_count {
                    max_count = count;
                    max_count_cell = (i as i32, j as i32);
                }
                new_boundary_cells.push(*cur);
            }
        }
    }
    // Update the frontier based on our new boundary cells
    for cur in &new_boundary_cells {
        let (i_, j_) = cur;
        let i = *i_; let j = *j_;
        for unk_neighbor in get_unknown_neighbors(&solve_state.solve_grid, i, j) {
            solve_state.frontier.insert(unk_neighbor);
        }
    }
    (max_count_cell, new_boundary_cells)
}

fn update_marked_cells() {

}

/// Recursively update likelihoods for this cell and its neighbors
/// Returns a set of all updated cells.
fn update_likelihoods(solve_state: &mut SolveState, i: usize, j: usize) -> HashSet<(usize, usize)> {
    let mut updated_cells = HashSet::with_capacity(8);
    update_likelihoods_impl(solve_state, i, j, &mut updated_cells);
    updated_cells
}

fn update_likelihoods_impl(solve_state: &mut SolveState, i: usize, j: usize, visited: &mut HashSet<(usize, usize)>) {
    // Step 1: if we've been here before, bail.
    let coords = (i, j);
    if visited.contains(&coords) {
        return;
    }
    // Then mark this cell as visited.
    visited.insert(coords);

    // Step 2: consider all the cell's neighbors for updates
    let count = get(&solve_state.count_grid, i, j).neighbor_mine_count;
    if count <= 0 {
        panic!("cannot update likelihoods for a cell with no neighboring mines: {i},{j}");
    }
    let neighbors = get_neighbors(&solve_state.solve_grid, i, j);

    let unknown_neighbors: Vec<&(usize, usize)> = neighbors.iter().filter(|(ni, nj)|
       get(&solve_state.solve_grid, *ni, *nj).is_unknown()
    ).collect();
    let n_unknown_neighbors = unknown_neighbors.len();

    let n_known_mine_neighbors: i32 = neighbors.iter().map(|(ni, nj)|
        if get(&solve_state.solve_grid, *ni, *nj).mine_likelihood == 1.0 { 1 } else { 0 }
    ).sum();

    // The likelihood that this cell's neighbor is a mine, based off of this cell's count:
    // Number of remaining mines bordering this cell / number of unknown neighbors of this cell
    let n_remaining_mine_neighbors = count - n_known_mine_neighbors;
    let mine_likelihood = n_remaining_mine_neighbors as f64 / n_unknown_neighbors as f64;

    // Update our unknown neighbors' likelihoods
    let mut updated = false;
    // for_each_neighbor(&solve_state.solve_grid, i, j, &mut |mut cell, ni, nj| {
    for (ni_, nj_) in &unknown_neighbors {
        let ni = *ni_; let nj = *nj_;
        let mut neighbor = get_mut(&mut solve_state.solve_grid, ni, nj);
        // Update this neighbor if the mine_likelihood based off of this cell is greater than the
        // cell's current mine likelihood, OR if the mine likelihood from this cell is zero.
        if mine_likelihood > neighbor.mine_likelihood {
            updated = true;
            neighbor.mine_likelihood = mine_likelihood;
            neighbor.leader = (i as i32, j as i32); // The leader now the current cell.
        }
    }

    // If we didn't update anything, terminate the search, since our neighbors won't have been
    // update either.
    // If this was a recently marked cell that made an impact, than we _should_ have updated
    // something. If it was a recently marked cell that didn't make an impact, then
    // we should start updating again from somewhere else?
    // Or not??? Idea here is to avoid exploring neighbors that are unimpacted
    if !updated {
        return;
    }
    // Otherwise, recurse on our neighboring boundary cells, since we may have updated their
    // probabilities.
    // let boundary_neighbors: Vec<(usize, usize)> = neighbors.iter().filter(|(ni, nj)|
    for (ni_, nj_) in &unknown_neighbors {
        let ni = *ni_; let nj = *nj_;
        if solve_state.is_boundary_cell(ni, nj) {
            update_likelihoods_impl(solve_state, ni, nj, visited);
        }
    }
}

fn get_unknown_neighbors(solve_grid: &SolveGrid, i: usize, j: usize) -> Vec<(usize, usize)>{
    let mut unknown_neighbors = Vec::with_capacity(8);
    for_each_neighbor(solve_grid, i, j, &mut |cell, ni, nj| {
        if cell.is_unknown() {
            unknown_neighbors.push((i, j))
        }
    });
    unknown_neighbors
}

fn get_num_mine_and_unknown_neighbors(solve_grid: &SolveGrid, i: usize, j: usize) -> (usize, usize) {
    let mut n_unknown_neighbors: usize = 0;
    let mut n_mine_neighbors: usize = 0;
    for_each_neighbor(solve_grid, i, j, &mut |cell, ni, nj| {
        if cell.mine_likelihood == 1.0 {
            n_mine_neighbors += 1;
        } else if cell.mine_likelihood != 0.0 {
            n_unknown_neighbors += 1;
        }
    });
    (n_mine_neighbors, n_unknown_neighbors)
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
        let new_boundary_cells = mark(&mut solve_state, 0, 2, &mine_map);
        println!("{}", count_grid::to_string(&solve_state.count_grid));
        println!("{}", solve_grid::to_string(&solve_state.solve_grid));
        println!("{:?}", new_boundary_cells);

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