use crate::solve_grid::*;
use crate::solve_grid;
use crate::count_grid;
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
    let mut solve_cell = get_mut(&mut solve_state.solve_grid, i, j);
    solve_cell.mine_likelihood = mine_likelihood;
    // If the cell is now known, remove it from the frontier (if it's on there, otherwise no-op.)
    if !solve_cell.is_unknown() {
        solve_state.frontier.remove(&(i, j));
    }
    // Now try updating the likelihoods and see if we arrive at a contradiction.
    match update_likelihoods_after_state_change(solve_state) {
        Ok(_) => false,
        Err(_) => true,
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assume;

    #[test]
    pub fn test_sim1() {
        assume::test_sim1();
    }

}