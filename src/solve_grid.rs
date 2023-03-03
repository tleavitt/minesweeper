use std::cmp::max;
use std::collections::HashSet;
use crate::count_grid::*;
use crate::mine_map::*;
use crate::count_grid;
use crate::grid::*;

#[derive(Debug, Clone)]
pub struct SolveCell {
    pub mine_likelihood: f64,  // The current estimated likelihood that this cell is a mine
    unknown_neighbor_mine_likelihood: f64,  // The likelihood that an unknown neighbor of this cell is a mine,
                                    // based of of this cell's mine count and known neighbors.
    // leader: (i32, i32) // The cell who's determining the likelihood of this one - must be a neighbor - necessary?
    // is_settled: bool // Whether all this cell's neighbors are known (i.e. known to be mines or known to not be mines)
}

impl SolveCell {
    pub fn is_unknown(&self) -> bool {
        self.mine_likelihood != 0.0 && self.mine_likelihood != 1.0
    }
    pub fn is_mine(&self) -> bool {
        self.mine_likelihood == 1.0
    }
}


// Internal board used by the solver
pub type SolveGrid = Vec<Vec<SolveCell>>;

pub fn flatten_cells(solve_grid: &SolveGrid) -> Vec<Vec<f64>> {
    solve_grid.iter().map(
        |row| row.iter().map(
            |cell| cell.mine_likelihood
        ).collect()
    ).collect()
}

// Container for data structures the solver uses.
#[derive(Debug, Clone)]
pub struct SolveState {
    pub solve_grid: SolveGrid,
    pub count_grid: CountGrid,
    pub nmines: usize,
    pub boundary: HashSet<(usize, usize)>, // cells with a non-zero mine count
    pub frontier: HashSet<(usize, usize)>, // unknown cells bordering boundary cells.
}

///
/// Terms:
/// unknown cell: a cell with a non-zero and non-one mine probability
/// known cell: a cell with a zero or one probability of being a mine
/// marked cell: a cell that's been marked by the user, revealing its mine count.
/// settled cell: a cell where none of it's neighbors are unknown.
/// The goal is to convert the entire board to known cells.
/// boundary cell: a marked cell with a non-zero mine count. Can be settled or unsettled.
/// frontier cell: an unknown cell that borders a boundary cell (i.e. a candidate for being a mine)
///
/// We track the boundary cells and the frontier cells globally.
impl SolveState {
    pub fn init(nrows: usize, ncols: usize, nmines: usize) -> SolveState {
        SolveState {
            solve_grid: init_solve_grid(nrows, ncols, nmines),
            count_grid: init_count_grid(nrows, ncols),
            nmines,
            boundary: HashSet::with_capacity(16),
            frontier: HashSet::with_capacity(16),
        }
    }


    // Returns whether the given cell is settled, i.e. has no unknown neighbors.
    pub fn is_settled(&self, i: usize, j: usize) -> bool {
        let (_, n_unknown_neighbors) = get_num_mine_and_unknown_neighbors(&self.solve_grid, i, j);
        n_unknown_neighbors == 0
    }

    ///
    /// Return the probability that an unknown neighbor of this cell is a mine, given
    /// this cell's mine count and known neighbors. Returns an error if there are not
    /// enough unknown neighbors for this cell.
    ///
    pub fn get_mine_neighbor_likelihood(&self, i: usize, j: usize) -> Result<f64, InvalidSolveCell> {
        let count_cell = get(&self.count_grid, i, j);
        // Only makes sense to compute this for marked cells.
        if !count_cell.is_marked() {
            panic!("Attempted to get mine neighbor likelihood of unmarked cell")
        }
        let (n_known_mine_neighbors, n_unknown_neighbors) = get_num_mine_and_unknown_neighbors(&self.solve_grid, i, j);
        // If all neighbors are known, then no unknown neighbors can be mines.
        if n_unknown_neighbors == 0 {
            // The number of known mine neighbors should equal the cell count - otherwise
            // the solve state is invalid.
            if n_known_mine_neighbors != count_cell.neighbor_mine_count as usize {
               return Err(InvalidSolveCell { i, j })
            }
            // Otherwise all is well, neighbor mine likelihood is zero.
            return Ok(0.0)
        }
        let neighbor_mine_count = count_cell.neighbor_mine_count  as usize;
        if n_known_mine_neighbors > neighbor_mine_count {
            return Err(InvalidSolveCell { i, j })
        }
        let n_remaining_mine_neighbors = count_cell.neighbor_mine_count as usize - n_known_mine_neighbors;

        // The likelihood that this cell's neighbor is a mine, based off of this cell's count:
        // Number of remaining mines bordering this cell / number of unknown neighbors of this cell
        // If there are more remaining mine neighbors than unknown neighbors, there's no space
        // left for the needed mines - the state is invalid.
        if n_remaining_mine_neighbors > n_unknown_neighbors {
            return Err(InvalidSolveCell { i, j })
        }
        Ok(n_remaining_mine_neighbors as f64 / n_unknown_neighbors as f64)
    }

    pub fn get_num_rows(&self) -> usize {
        get_num_rows(&self.count_grid)
    }
    pub fn get_num_cols(&self) -> usize {
        get_num_cols(&self.count_grid)
    }

    pub fn pretty_print(&self) -> String {
        let mut str = String::new();
        // First print a header
        str.push_str(&"   ");
        for j in 0..self.get_num_cols() {
            str.push_str(&*format!("{:02} ", j));
        }
        str.push('\n');
        for i in 0..self.get_num_rows() {
            str.push_str(&*format!("{:02} ", i));
            for j in 0..self.get_num_cols() {
                let count_cell = get(&self.count_grid, i, j);
                let sym: String = if count_cell.is_marked() {
                    count_cell.neighbor_mine_count.to_string()
                } else {
                    let solve_cell = get(&self.solve_grid, i, j);
                    String::from(if solve_cell.mine_likelihood == 1.0 {
                        "!"
                    } else {
                        "-"
                    })
                };
                str.push_str(&*format!(" {} ", sym));
            }
            str.push('\n');
        }
        str
    }
}

pub fn init_solve_grid(nrows: usize, ncols: usize, nmines: usize) -> SolveGrid {
    let mut solve_grid: SolveGrid = Vec::with_capacity(nrows);
    let prior: f64 = nmines as f64 / (nrows * ncols) as f64;
    for _ in 0..nrows {
        solve_grid.push(vec![SolveCell {mine_likelihood: prior, unknown_neighbor_mine_likelihood: -1.0 }; ncols ]);
    }
    solve_grid
}

///
/// Main entrypoint for updating solve state.
///
pub fn update_after_mark(solve_state: &mut SolveState, marked_cells: &Vec<(usize, usize)>) {
    apply_mark(solve_state, marked_cells);
    match update_likelihoods_after_state_change(solve_state) {
        Ok(_) => {}
        Err(e) => {
            panic!("Found invalid solve cell while updating likelihoods: {:?}", e)
        }
    };
}

///
/// Marks a cell, updating the count grid, the solve grid, the frontier, and the boundary.
/// Returns the new boundary cells revealed by marking.
///
pub fn apply_mark(solve_state: &mut SolveState, marked_cells: &Vec<(usize, usize)>) -> HashSet<(usize, usize)> {
    // First, set all marked cells to have zero likelihood and remove them from the frontier if present.
    for cur in marked_cells.iter() {
        let (i_, j_) = cur;
        // Marked cells are not mines, so set their likelihoods to zero.
        let mut cell = get_mut(&mut solve_state.solve_grid, *i_, *j_);
        cell.mine_likelihood = 0.0;
        // Remove the cell from the frontier if it was present.
        solve_state.frontier.remove(cur);
    }

    let mut new_boundary_cells: HashSet<(usize, usize)> = HashSet::with_capacity(marked_cells.len());
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
        if count_cell.is_boundary_cell() {
            new_boundary_cells.insert(*cur);
        }
    }
    // Update the boundary and frontier based on our new boundary cells
    for cur in &new_boundary_cells {
        solve_state.boundary.insert(*cur);
        let (i_, j_) = cur;
        let i = *i_; let j = *j_;
        for unk_neighbor in get_unknown_neighbors(&solve_state.solve_grid, i, j) {
            // println!("unk neighbor: {:?}", unk_neighbor);
            solve_state.frontier.insert(unk_neighbor);
        }
    }
    new_boundary_cells
}

///
/// Updates the solve_state probabilities after some change to the solve state (e.g. a cell was marked,
/// or we simulated a cell probability being set).
///
pub fn update_likelihoods_after_state_change(solve_state: &mut SolveState) -> Result<(), InvalidSolveCell> {
    // Repeat until we reach stability
    loop {
        // Phase 1: update the mine_neighbor_likelihood for all boundary cells.
        // TODO: instead of running on all cells, could run only on cells that need updates.
        //  Need to determine which cells these are. Might be tricky
        let updated = update_unknown_neighbor_mine_likelihoods(solve_state)?;
        if !updated {
            break;
        }
        // Phase 2: update the frontier cell
        let updated = update_frontier_likelihoods(solve_state)?;
        if !updated {
            break;
        }
        update_frontier(solve_state);
        update_interior(solve_state)?;
    }
    Ok(())
}

/// Error class for a state contradiction
/// Indicates that the provided cell has a mine_likelihood of both 0.0 and 1.0
#[derive(Debug, Clone)]
pub struct InvalidSolveCell {
    i: usize,
    j: usize,
}

/// Update the mine neighbor likelihoods of all boundary cells.
/// Returns whether we updated the likelihood of any cell.
fn update_unknown_neighbor_mine_likelihoods(solve_state: &mut SolveState) -> Result<bool, InvalidSolveCell> {
    let mut updated = false;
    // TODO: only iterate over boundary cells who had a state change/whose neighbors had a state
    // change?
    for (i_, j_) in &solve_state.boundary {
        let (i, j) = (*i_, *j_);
        let new_likelihood = solve_state.get_mine_neighbor_likelihood(i, j)?;
        let mut boundary_cell = get_mut(&mut solve_state.solve_grid, i, j);
        if boundary_cell.unknown_neighbor_mine_likelihood != new_likelihood {
            updated = true;
            boundary_cell.unknown_neighbor_mine_likelihood = new_likelihood;
        }
    }
    Ok(updated)
}

/// Update the mine likelihoods of all frontier cells.
/// Returns whether we updated the likelihood of any cell.
fn update_frontier_likelihoods(solve_state: &mut SolveState) -> Result<bool, InvalidSolveCell> {
    let mut updated = false;

    for (i_, j_) in &solve_state.frontier {
        let (i, j) = (*i_, *j_);
        // Invariant: cells on the frontier must be unknown
        let frontier_cell = get(&solve_state.solve_grid, i, j);
        if !frontier_cell.is_unknown() {
            panic!("Known cell on frontier: {:?}", frontier_cell);
        }

        // The new mine likelihood for this cell is EITHER:
        // 0 if any of the boundary neighbors have a neighbor likelihood of 0, else
        // 1 if any of the boundary neighbors have a neighbor likelihood of 1, else
        // the maximum of all boundary neighbor likelihoods.
        // IF one boundary neighbor has a neighbor likelihood of 0 and another has a neighbor
        // likelihood of 1, the state is invalid and we return an error immediately.

        let mut cur_mine_likelihood: f64 = -1.0; // Current computed value for the mine likelihood of this cell (i, j)
        let boundary_neighbor_likelihoods: Vec<f64> = get_boundary_neighbors(solve_state, i, j).iter()
            .map(|(ni, nj)| get(&solve_state.solve_grid, *ni, *nj).unknown_neighbor_mine_likelihood)
            .collect();

        for neighbor_likelihood in boundary_neighbor_likelihoods {
            // Sort the current likelihood and the neighbor likelihood for consistency
            let (smaller, larger) = if cur_mine_likelihood <= neighbor_likelihood {
                (cur_mine_likelihood, neighbor_likelihood)
            } else {
                (neighbor_likelihood, cur_mine_likelihood)
            };
            if smaller == 0.0 {
                // A contradiction occurs if the smaller likelihood is zero and the larger is one.
                // Return an error right away.
                if larger == 1.0 {
                    return Err(InvalidSolveCell{ i, j })
                }
                // Propagate the zero immediately.
                cur_mine_likelihood = smaller;
            } else {
                // Propagate the larger probability if none are zero.
                cur_mine_likelihood = larger
            }
        }

        // Sanity check: if no boundary neighbors for this cell, panic
        if cur_mine_likelihood < 0.0 {
            panic!("No boundary neighbors for frontier cell {i},{j}");
        }

        let mut frontier_cell = get_mut(&mut solve_state.solve_grid, i, j);
        if frontier_cell.mine_likelihood != cur_mine_likelihood {
            updated = true;
            frontier_cell.mine_likelihood = cur_mine_likelihood;
        }
    }
    Ok(updated)
}

///
/// Check our frontier cells and remove known cells (i.e. mines and known non-mines).
///
fn update_frontier(solve_state: &mut SolveState) {
    let mut new_known_cells: Vec<(usize, usize)> = Vec::with_capacity(solve_state.nmines);
    for coord in &solve_state.frontier {
        let (i_, j_) = coord;
        let i = *i_; let j = *j_;
        if !get(&solve_state.solve_grid, i, j).is_unknown() {
            new_known_cells.push(*coord);
        }
    }
    for coords in new_known_cells {
        solve_state.frontier.remove(&coords);
    }
}

///
/// Updates the likelihood for interior cells, i.e. unknown cells that are not on the frontier.
///
fn update_interior(solve_state: &mut SolveState) -> Result<(), InvalidSolveCell> {
    let mut known_mine_count: usize = 0;
    let mut unknown_cell_count: usize = 0;

    let nrows = solve_state.get_num_rows();
    let ncols = solve_state.get_num_cols();
    for i in 0..nrows {
        for j in 0..ncols {
            let cell = get(&solve_state.solve_grid, i, j);
            if cell.is_unknown() {
                unknown_cell_count += 1
            } else if cell.is_mine() {
                known_mine_count += 1
            }
        }
    }
    // If there are more known mines than total mines, something is wrong - return an error
    if known_mine_count > solve_state.nmines {
        return Err(InvalidSolveCell { i: 0, j: 0}) // TODO: not really the right coordinates, but whatever.
    }

    let interior_likelihood = (solve_state.nmines - known_mine_count) as f64 / unknown_cell_count as f64;
    for i in 0..nrows {
        for j in 0..ncols {
            // An interior cell is unknown and not on the frontier
            if get(&solve_state.solve_grid, i, j).is_unknown()
                && !solve_state.frontier.contains(&(i, j)) {
                // Sanity check
                for_each_neighbor(&solve_state.solve_grid, i, j, &mut |cell, ni, nj| {
                    if get(&solve_state.count_grid, ni, nj).is_marked() {
                        panic!("Unknown cell with marked neighbor is not on the frontier: {ni},{nj}");
                    }
                });
                // Update the interior likelihood based on the mine count.
                get_mut(&mut solve_state.solve_grid, i, j).mine_likelihood = interior_likelihood;
            }
        }
    }
    Ok(())
}

///
/// Get an unmarked cell with the lowest likelihood of being a mine.
///
pub fn get_least_likely_unknown_cell(solve_state: &SolveState) -> Option<((usize, usize), f64)> {
    // Simply choose the unmarked cell with the lowest likelihood of bieng a mine.
    // TODO: implement lookahead by simulating mines.
    // Iterate over the whole board looking for the best unmarked cell.
    let mut best_likelihood: f64 = 1.0;
    let mut maybe_best_cell:Option<(usize, usize)> = None;
    for i in 0..solve_state.get_num_rows() {
        for j in 0..solve_state.get_num_cols() {
            if !get(&solve_state.count_grid, i, j).is_marked() {
                if get(&solve_state.solve_grid, i, j).mine_likelihood < best_likelihood {
                    best_likelihood = get(&solve_state.solve_grid, i, j).mine_likelihood;
                    maybe_best_cell = Some((i, j));
                }
            }
        }
    }
    maybe_best_cell.map(|best_cell| (best_cell, best_likelihood))
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

fn get_boundary_neighbors(solve_state: &SolveState, i: usize, j: usize) -> Vec<(usize, usize)>{
    let mut boundary_neighbors = Vec::with_capacity(8);
    for_each_neighbor(&solve_state.count_grid, i, j, &mut |cell, ni, nj| {
        if cell.is_boundary_cell() {
            boundary_neighbors.push((ni, nj))
        }
    });
    boundary_neighbors
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
fn make_move_and_print(m: (usize, usize), solve_state: &mut SolveState, mine_map: &MineMap) {
    let marked_cells = count_grid::mark(&mut solve_state.count_grid, m.0, m.1, mine_map);
    println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
    update_after_mark(solve_state, &marked_cells);
    println!("solve_grid: {}", to_string(&solve_state.solve_grid));
    println!("{:?}", solve_state);
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::mine_map::MineMap;
    use crate::solve_grid;

    #[test]
    pub fn test_update_likelihoods1()  {
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
        println!("solve_grid: {}", to_string(&solve_state.solve_grid));

        let marked_cells = count_grid::mark(&mut solve_state.count_grid, 0, 2, &mine_map);
        let new_boundary_cells = apply_mark(&mut solve_state, &marked_cells);
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", to_string(&solve_state.solve_grid));
        println!("new boundary cells: {:?}", new_boundary_cells);

        update_likelihoods_after_state_change(&mut solve_state).expect("Invalid cell state!");
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", to_string(&solve_state.solve_grid));

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
            flatten_cells(&solve_state.solve_grid)
        );
        println!("{:?}", solve_state);
        assert_eq!(None, get_least_likely_unknown_cell(&solve_state));


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
            flatten_cells(&solve_state.solve_grid)
        );
        println!("{:?}", solve_state);
    }


    #[test]
    pub fn test_update_likelihoods2() {
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

        update_likelihoods_after_state_change(&mut solve_state).expect("Invalid cell state!");
        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", solve_grid::to_string(&solve_state.solve_grid));

    }

    #[test]
    pub fn test_update_likelihoods3() {
        let mine_map: MineMap = vec![
            vec![true,  false, false, false, false],
            vec![false, false, false, false, false],
            vec![true,  true,  false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, true,  true,  false],
        ];
        let mut solve_state = SolveState::init(
            get_num_rows(&mine_map),
            get_num_cols(&mine_map),
            5
        );

        println!("count_grid: {}", count_grid::to_string(&solve_state.count_grid));
        println!("solve_grid: {}", to_string(&solve_state.solve_grid));
        println!("{:?}", &solve_state);

        make_move_and_print((1, 1), &mut solve_state, &mine_map);
        assert_eq!(vec![
            vec![-1,  -1, -1, -1, -1],
            vec![-1,   3, -1, -1, -1],
            vec![-1,  -1, -1, -1, -1],
            vec![-1,  -1, -1, -1, -1],
            vec![-1,  -1, -1, -1, -1]
        ],
                   count_grid::flatten_cells(&solve_state.count_grid)
        );
        // interior likelihood
        let il = 5 as f64 / 24 as f64;
        assert_eq!(vec![
            // Without interior adjustments
            // vec![0.375, 0.375, 0.375, 0.2, 0.2],
            // vec![0.375, 0.0,   0.375, 0.2, 0.2],
            // vec![0.375, 0.375, 0.375, 0.2, 0.2],
            // vec![0.2, 0.2, 0.2, 0.2, 0.2],
            // vec![0.2, 0.2, 0.2, 0.2, 0.2],
            vec![0.375, 0.375, 0.375,  il,  il],
            vec![0.375, 0.0,   0.375,  il,  il],
            vec![0.375, 0.375, 0.375,  il,  il],
            vec![ il,  il,  il,  il,  il],
            vec![ il,  il,  il,  il,  il],
        ],
                   flatten_cells(&solve_state.solve_grid)
        );

        make_move_and_print((0, 3), &mut solve_state, &mine_map);
        make_move_and_print((3, 0), &mut solve_state, &mine_map);
        assert_eq!(vec![
            vec![-1,   1,  0,  0,  0],
            vec![-1,   3,  1,  0,  0],
            vec![-1,  -1,  1,  0,  0],
            vec![ 2,  -1,  3,  2,  1],
            vec![-1,  -1, -1, -1, -1]],
                   count_grid::flatten_cells(&solve_state.count_grid)
        );
        assert_eq!(vec![
            vec![2.0/3.0, 0.0, 0.0, 0.0, 0.0],
            vec![2.0/3.0, 0.0,   0.0, 0.0, 0.0],
            vec![2.0/3.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![1.0/3.0, 2.0/3.0, 2.0/3.0, 2.0/3.0, 2.0/3.0],],
                   flatten_cells(&solve_state.solve_grid)
        );
        assert_eq!(Some(((3, 1), 0.0)), get_least_likely_unknown_cell(&solve_state));

        make_move_and_print((4, 0), &mut solve_state, &mine_map);
        assert_eq!(vec![
            vec![-1,   1,  0,  0,  0],
            vec![-1,   3,  1,  0,  0],
            vec![-1,  -1,  1,  0,  0],
            vec![ 2,   3,  3,  2,  1],
            vec![ 0,   1, -1, -1, -1]],
                   count_grid::flatten_cells(&solve_state.count_grid)
        );
        assert_eq!(vec![
            vec![0.5, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.0, 0.0, 0.0],
            vec![1.0, 1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0, 0.0],],
                   flatten_cells(&solve_state.solve_grid)
        );
    }

    #[test]
    pub fn test_clone() {
        let mine_map: MineMap = vec![
            vec![true,  false, false, false, false],
            vec![false, false, false, false, false],
            vec![true,  true,  false, false, false],
            vec![false, false, false, false, false],
            vec![false, false, true,  true,  false],
        ];
        let mut solve_state = SolveState::init(
            get_num_rows(&mine_map),
            get_num_cols(&mine_map),
            5
        );
        let marked_cells = count_grid::mark(&mut solve_state.count_grid, 0, 4, &mine_map);
        solve_grid::update_after_mark(&mut solve_state, &marked_cells);

        let mut cloned_state = solve_state.clone();
        // assert!(cloned_state.solve_grid.eq(&solve_state.solve_grid));
        // assert!(cloned_state.count_grid.eq(&solve_state.count_grid));
        assert_eq!(cloned_state.boundary, solve_state.boundary);
        assert_eq!(cloned_state.frontier, solve_state.frontier);
        println!("Original: {}", count_grid::to_string(&solve_state.count_grid));
        println!("Original frontier:\n{:?}", solve_state.frontier);
        println!("Clone: {}", count_grid::to_string(&cloned_state.count_grid));
        println!("Clone frontier:\n{:?}", cloned_state.frontier);

        let marked_cells = count_grid::mark(&mut cloned_state.count_grid, 4, 0, &mine_map);
        solve_grid::update_after_mark(&mut cloned_state, &marked_cells);

        // assert!(!cloned_state.solve_grid.eq(&solve_state.solve_grid));
        // assert!(!cloned_state.count_grid.eq(&solve_state.count_grid));
        assert_ne!(cloned_state.boundary, solve_state.boundary);
        assert_ne!(cloned_state.frontier, solve_state.frontier);
        println!("Original: {}", count_grid::to_string(&solve_state.count_grid));
        println!("Original frontier:\n{:?}", solve_state.frontier);
        println!("Clone: {}", count_grid::to_string(&cloned_state.count_grid));
        println!("Clone frontier:\n{:?}", cloned_state.frontier);
    }

}