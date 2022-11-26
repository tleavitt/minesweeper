use crate::solve_grid::*;
use crate::solve_grid;
use crate::count_grid;
use crate::mine_map::*;
use crate::grid::*;

pub fn yields_contradiction(
    parent: &SolveState, i: usize, j: usize, mine_likelihoo: f64) -> bool {
    // Clone the solve state so we can mess with it without affecting the parent.
    let mut sim_state: SolveState = parent.clone();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

}