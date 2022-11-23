pub struct SolveCell {
    mine_likelihood: f32, // The probability that this cell is a mine
    adj_count: i32,       // The number of mines adjacent to this one. -1 means the square is unknown
}

impl SolveCell {
    fn is_marked(&self) -> bool {
        return self.adj_count != -1
    }
}

// Internal board used by the solver
type SolveBoard = Vec<Vec<SolveCell>>;