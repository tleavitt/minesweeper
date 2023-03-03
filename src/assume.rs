use crate::solve_grid::*;
use crate::solve_grid;
use crate::mine_map::*;
use crate::grid::*;

///
/// Manually set the mine likelihood of cell i,j to the given mine_likelihood, and return
/// whether doing so creates a contradiction in the solve state. If a contradiction is reached,
/// we know the solve state is invalid - if the solve state was valid _before_ manually setting
/// the likelihood of i,j, then we know that the probability we set must be wrong.
///
pub fn yields_contradiction(
    solve_state: &mut SolveState, i: usize, j: usize, mine_likelihood: f64) -> bool {
    // Safety belts: don't update mine_likelihood on a marked cell.
    // We'll allow you to update the likelihood on a marked, known cell, but
    // that probably won't be useful.
    if get(&solve_state.count_grid, i, j).is_marked() {
        panic!("Cell being examined for contradictions is marked: {i},{j}");
    }

    // Manually update the mine_likelihood of the given cell
    apply_manual_update(solve_state, i, j, mine_likelihood);
    // Now try updating the likelihoods and see if we arrive at a contradiction.
    match update_likelihoods_after_state_change(solve_state) {
        Ok(_) => false,
        Err(_) => true,
    }
}

fn apply_manual_update(solve_state: &mut SolveState, i: usize, j: usize, mine_likelihood: f64) {
    // Manually update the mine_likelihood of the given cell
    let mut solve_cell = get_mut(&mut solve_state.solve_grid, i, j);
    solve_cell.mine_likelihood = mine_likelihood;
    // If the cell is now known, remove it from the frontier (if it's on there, otherwise no-op.)
    if !solve_cell.is_unknown() {
        solve_state.frontier.remove(&(i, j));
    }
}

///
/// Simulate setting the mine likelihood of cell i,j, and return whether doing so creates an invalid
/// solve state.
///
pub fn simulate_cell_update(solve_state: &SolveState, i: usize, j: usize, mine_likelihood: f64) -> bool {
    let mut sim_state = solve_state.clone();
    yields_contradiction(&mut sim_state, i, j, mine_likelihood)
}

fn get_num_boundary_neighbors(solve_state: &SolveState, i: usize, j: usize) -> usize {
    let mut n: usize = 0;
    for_each_neighbor(&solve_state.count_grid, i, j, &mut |cell, ni, nj| {
        if cell.is_boundary_cell() {
            n += 1
        }
    });
    n
}

///
/// Ranks all frontier cells by the number of boundary neighbors they have.
///
pub fn get_frontier_cells_ranked_by_n_boundary_neighbors(solve_state: &SolveState) -> Vec<(usize, usize)> {
    let mut frontier_and_count: Vec<(&(usize, usize), usize)> = solve_state.frontier.iter().map(
        |coord| {
            (coord, get_num_boundary_neighbors(solve_state, coord.0, coord.1))
        }
    ).collect();
    // Sort in reverse order - largest to smallest
    frontier_and_count.sort_by(|cc1, cc2| cc2.1.cmp(&cc1.1));
    // frontier_and_count.sort_by(|cc1, cc2| cc1.1.cmp(&cc2.1));
    frontier_and_count.iter().map(|cell_and_count| *(cell_and_count.0)).collect()
}

///
/// Search the solve grid for an unknown cell that we can determine must be a mine/not a mine
/// by assuming the opposite and deriving a contradiction. Returns the first such cell
/// we can find, and it's determined mine probability (1 or 0).
/// TODO: cache search state? To avoid re-simulating cells when their state hasn't changed at all.
///
pub fn search_for_contradictions(solve_state: &SolveState) -> Option<((usize, usize), f64)>{
    let candidates = get_frontier_cells_ranked_by_n_boundary_neighbors(solve_state);
    for (i, j) in candidates {
        // First try assuming it's a mine. This is more useful since it gives us a cell to mark
        // immediately.
        if simulate_cell_update(solve_state, i, j, 1.0) {
            // Assuming it was a mine led to contradiction, so it must not be a mine.
            return Some(((i, j), 0.0))
        } else if simulate_cell_update(solve_state, i, j, 0.0) {
            // Assuming it was not a mine led to contradiction, so it must be a mine.
            return Some(((i, j), 1.0))
        }
    }
    // Couldn't find any immediate contradictions, return none.
    None
}

pub fn update_from_contradiction(solve_state: &mut SolveState, res: ((usize, usize), f64)) {
    let ((i, j), mine_likelihood) = res;
    apply_manual_update(solve_state, i, j, mine_likelihood);
    update_likelihoods_after_state_change(solve_state).expect("Invalid solve state");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assume;
    use crate::count_grid;

    #[test]
    pub fn test_sim1() {
        let mine_map: MineMap = vec![
            vec![false, true ,  false,  true , false],
            vec![false, false,  false,  false, false],
        ];
        let mut solve_state = SolveState::init(
            get_num_rows(&mine_map),
            get_num_cols(&mine_map),
            2
        );
    
        let marked_cells1 = count_grid::mark(&mut solve_state.count_grid, 1, 1, &mine_map);
        let marked_cells2 = count_grid::mark(&mut solve_state.count_grid, 1, 2, &mine_map);
        let marked_cells3 = count_grid::mark(&mut solve_state.count_grid, 1, 3, &mine_map);
        update_after_mark(&mut solve_state, &([marked_cells1, marked_cells2, marked_cells3].concat()));
        println!("{}", solve_grid::to_string(&solve_state.solve_grid));
        // Looks like:
        // - - - - -
        // - 1 2 1 -
        // From this we can infer that the top middle cell is not a mine -
        // If it were a mine, then both the ones would be settled, and there would be no place
        // left for the second mine for the 2.
        assert_eq!(true, simulate_cell_update(&solve_state, 0, 2, 1.0));
        assert_eq!(false, simulate_cell_update(&solve_state, 0, 0, 0.0));
        assert_eq!(true, simulate_cell_update(&solve_state, 0, 0, 1.0));
        assert_eq!(Some(((0, 2), 0.0)), search_for_contradictions(&solve_state));    }

    #[test]
    pub fn test_sim2() {
        let mine_map: MineMap = vec![
            vec![false, true, true, false],
            vec![false, false, false, false],
        ];
        let mut solve_state = SolveState::init(
            get_num_rows(&mine_map),
            get_num_cols(&mine_map),
            2
        );

        let marked_cells1 = count_grid::mark(&mut solve_state.count_grid, 1, 0, &mine_map);
        let marked_cells2 = count_grid::mark(&mut solve_state.count_grid, 1, 1, &mine_map);
        let marked_cells3 = count_grid::mark(&mut solve_state.count_grid, 1, 2, &mine_map);
        let marked_cells4 = count_grid::mark(&mut solve_state.count_grid, 1, 3, &mine_map);
        update_after_mark(&mut solve_state, &([marked_cells1, marked_cells2, marked_cells3, marked_cells4].concat()));
        println!("{}", solve_grid::to_string(&solve_state.solve_grid));
        // Looks like:
        // - - - -
        // 1 2 2 1
        // The middle two cells must be mines:
        // If one of them isn't, then the corner and the other middle cell must be mines
        // because of the 2, and then the other corner must be a mine because of the other 2,
        // and then the 1 in the second corner is bordering two mines.
        assert_eq!(true, simulate_cell_update(&solve_state, 0, 0, 1.0));
        assert_eq!(false, simulate_cell_update(&solve_state, 0, 0, 0.0));

        assert_eq!(true, simulate_cell_update(&solve_state, 0, 1, 0.0));
        assert_eq!(false, simulate_cell_update(&solve_state, 0, 1, 1.0));

        assert_eq!(true, simulate_cell_update(&solve_state, 0, 2, 0.0));
        assert_eq!(false, simulate_cell_update(&solve_state, 0, 2, 1.0));

        assert_eq!(true, simulate_cell_update(&solve_state, 0, 3, 1.0));
        assert_eq!(false, simulate_cell_update(&solve_state, 0, 3, 0.0));

        assert!(search_for_contradictions(&solve_state).is_some());
    }

}