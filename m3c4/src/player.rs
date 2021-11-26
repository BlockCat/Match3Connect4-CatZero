#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    Player1,
    Player2,
}

impl Player {
    pub fn next_player(&self) -> Self {
        match self {
            Player::Player1 => Player::Player2,
            Player::Player2 => Player::Player1,
        }
    }
}

impl Default for Player {
    fn default() -> Self {
        Player::Player1
    }
}
