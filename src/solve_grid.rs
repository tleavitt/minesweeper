pub struct SolveCell {
    mine_likelihood: f32, // The probability that this cell is a mine
}

// Internal board used by the solver
type SolveGrid = Vec<Vec<SolveCell>>;