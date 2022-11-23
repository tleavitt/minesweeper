pub struct SolveCell {
    mine_likelihood: f32, // The probability that this cell is a mine
}

const UNKNOWN_SOLVE_CELL: i32 = -1;

impl SolveCell {
    fn is_marked(&self) -> bool {
        return self.adj_count != UNKNOWN_SOLVE_CELL
    }
}

// Internal board used by the solver
type SolveGrid = Vec<Vec<SolveCell>>;