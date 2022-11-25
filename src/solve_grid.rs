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

pub fn flatten_cells(count_grid: &SolveGrid) -> Vec<Vec<f64>> {
    count_grid.iter().map(
        |row| row.iter().map(
            |cell| cell.mine_likelihood
        ).collect()
    ).collect()
}

// Container for data structures the solver uses.
#[derive(Debug)]
pub struct SolveState {
    solve_grid: SolveGrid,
    count_grid: CountGrid,
    nmines: usize,
    mines_found: usize,
    frontier: HashSet<(usize, usize)> // unknown cells bordering boundary cells.
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
/// Main entrypoint for updating solve state.
///
pub fn update_after_mark(solve_state: &mut SolveState, marked_cells: &Vec<(usize, usize)>) {
    let new_boundary_cells = apply_mark(solve_state, marked_cells);
    update_likelihoods_after_mark(solve_state, new_boundary_cells);
    update_metadata_after_mark(solve_state);
}

///
/// Marks a cell, updating the count grid, the solve grid, and the frontier,
/// and returning a list of new boundary cells revealed by the mark
///
pub fn apply_mark(solve_state: &mut SolveState, marked_cells: &Vec<(usize, usize)>) -> HashSet<(usize, usize)> {
    let mut new_boundary_cells: HashSet<(usize, usize)> = HashSet::with_capacity(marked_cells.len());

    // First, set all marked cells to have zero likelihood and remove them from the frontier if present.
    for cur in marked_cells.iter() {
        let (i_, j_) = cur;
        // Marked cells are not mines, so set their likelihoods to zero.
        let mut cell = get_mut(&mut solve_state.solve_grid, *i_, *j_);
        cell.mine_likelihood = 0.0;
        cell.leader = (-1, -1); // No leader for marked cells.
        // Remove the cell from the frontier if it was present.
        solve_state.frontier.remove(cur);
    }

    for cur in marked_cells.iter() {
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
                new_boundary_cells.insert(*cur);
            }
        }
    }
    // Update the frontier based on our new boundary cells
    for cur in &new_boundary_cells {
        println!("Boundary cell: {:?}", cur);
        let (i_, j_) = cur;
        let i = *i_; let j = *j_;
        for unk_neighbor in get_unknown_neighbors(&solve_state.solve_grid, i, j) {
            println!("unk neighbor: {:?}", unk_neighbor);
            solve_state.frontier.insert(unk_neighbor);
        }
    }
    new_boundary_cells
}

///
/// Updates the solve_state probabilities after marking a cell.
///
fn update_likelihoods_after_mark(solve_state: &mut SolveState, new_boundary_cells: HashSet<(usize, usize)>) {
    let mut to_process: HashSet<(usize, usize)> = new_boundary_cells;
    while !to_process.is_empty() {
        let (max_count_i, max_count_j) = to_process.iter().max_by_key(
            |(i, j)| get(&solve_state.count_grid, *i, *j).neighbor_mine_count).unwrap();

        let updated_cells = update_local_likelihoods(solve_state, *max_count_i, *max_count_j);

        for cell in updated_cells {
            to_process.remove(&cell);
        }
    }
}

///
/// Check our frontier cells and remove known mines.
///
fn update_metadata_after_mark(solve_state: &mut SolveState) {
    let mut new_mines: Vec<(usize, usize)> = Vec::with_capacity(solve_state.nmines);
    for coord in &solve_state.frontier {
        let (i_, j_) = coord;
        let i = *i_; let j = *j_;
        if get(&solve_state.solve_grid, i, j).mine_likelihood == 1.0 {
            solve_state.mines_found += 1;
            new_mines.push((i, j));
        }
    }
    for coords in new_mines {
        solve_state.frontier.remove(&coords);
    }
}

///
/// Given a fully expanded solve state, choose the next cell to mark.
///
fn choose_next_mark(solve_state: &SolveState) -> Option<(usize, usize)> {
    // Simply choose the frontier cell with lowest likelihood of being a mine
    // TODO: implement lookahead by simulating mines.
    solve_state.frontier.iter()
        .min_by(|(i1, j1), (i2, j2)| {
            get(&solve_state.solve_grid, *i1 ,*j1).mine_likelihood
                .partial_cmp(
                    &get(&solve_state.solve_grid, *i2, *j2).mine_likelihood
                ).unwrap()
        })
        .map(|b| *b)
}

/// Recursively update likelihoods for this cell and its neighbors
/// Returns a set of all updated cells.
fn update_local_likelihoods(solve_state: &mut SolveState, i: usize, j: usize) -> HashSet<(usize, usize)> {
    let mut updated_cells = HashSet::with_capacity(8);
    update_local_likelihoods_impl(solve_state, i, j, &mut updated_cells);
    updated_cells
}

fn update_local_likelihoods_impl(solve_state: &mut SolveState, i: usize, j: usize, visited: &mut HashSet<(usize, usize)>) {
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
        // TODO: might need to change this? why only go strictly greater?
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
            update_local_likelihoods_impl(solve_state, ni, nj, visited);
        }
    }
}

fn get_unknown_neighbors(solve_grid: &SolveGrid, i: usize, j: usize) -> Vec<(usize, usize)>{
    let mut unknown_neighbors = Vec::with_capacity(8);
    for_each_neighbor(solve_grid, i, j, &mut |cell, ni, nj| {
        if cell.is_unknown() {
            unknown_neighbors.push((ni, nj))
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

        let marked_cells = count_grid::mark(&mut solve_state.count_grid, 0, 2, &mine_map);
        let new_boundary_cells = apply_mark(&mut solve_state, &marked_cells);
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", solve_grid::to_string(&solve_state.solve_grid));
        println!("new boundary cells: {:?}", new_boundary_cells);

        update_likelihoods_after_mark(&mut solve_state, new_boundary_cells);
        update_metadata_after_mark(&mut solve_state);
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", solve_grid::to_string(&solve_state.solve_grid));

        assert_eq!(
            vec![
                vec![-1,  1,  0,  1, -1],
                vec![ 1,  1,  0,  1,  1],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  0,  0],
                vec![ 0,  0,  0,  1,  1],
                vec![ 0,  0,  0,  1, -1],
            ],
            count_grid::flatten_cells(&solve_state.count_grid)
        );

        assert_eq!(
            vec![
                vec![1.0, 0.0, 0.0, 0.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 1.0],
            ],
            solve_grid::flatten_cells(&solve_state.solve_grid)
        );
        println!("{:?}", solve_state);
        assert_eq!(None, choose_next_mark(&solve_state));
    }

    #[test]
    fn test_update_likelihoods2() {
        let mine_map: MineMap = vec![
            vec![true,  true,  true],
            vec![false, false, false],
        ];
        let mut solve_state = SolveState::init(
            get_num_rows(&mine_map),
            get_num_cols(&mine_map),
            3
        );
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", solve_grid::to_string(&solve_state.solve_grid));

        let marked_cells1 = count_grid::mark(&mut solve_state.count_grid, 1, 0, &mine_map);
        let marked_cells2 = count_grid::mark(&mut solve_state.count_grid, 1, 1, &mine_map);
        let marked_cells3 = count_grid::mark(&mut solve_state.count_grid, 1, 2, &mine_map);
        let new_boundary_cells = apply_mark(&mut solve_state, &([marked_cells1, marked_cells2, marked_cells3].concat()));
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", solve_grid::to_string(&solve_state.solve_grid));
        println!("new boundary cells: {:?}", new_boundary_cells);

        update_likelihoods_after_mark(&mut solve_state, new_boundary_cells);
        update_metadata_after_mark(&mut solve_state);
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", solve_grid::to_string(&solve_state.solve_grid));

    }
}