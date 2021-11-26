use std::{cmp::Reverse, collections::HashSet, fmt::Display};

use crate::{
    action::{BoardAction, Coordinate},
    player::Player,
};

pub const WIDTH: usize = 8;
pub const HEIGHT: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum Cell {
    Empty,
    Filled(Player),
}

impl Default for Cell {
    fn default() -> Self {
        Cell::Empty
    }
}

#[derive(Debug, Clone, Hash)]
pub enum TerminalResult {
    None,
    Win(Player),
    Draw,
}

impl Default for TerminalResult {
    fn default() -> Self {
        TerminalResult::None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MoveResult {
    Winner(Player),
    Draw,
    Three(Player),
}

#[derive(Debug, Default, Clone, Hash)]
pub struct Board {
    board: [[Cell; HEIGHT]; WIDTH],
}

impl From<[&str; 8]> for Board {
    fn from(a: [&str; 8]) -> Self {
        let mut board = Self::default();
        let a = a
            .into_iter()
            .map(|s| s.chars().collect::<Vec<_>>())
            .collect::<Vec<_>>();

        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                let cell = match a[y][x] {
                    'X' => Cell::Filled(Player::Player1),
                    'O' => Cell::Filled(Player::Player2),
                    ' ' => Cell::Empty,
                    _ => unreachable!(),
                };

                board.set(cell, Coordinate::new(x as isize, (HEIGHT - 1 - y) as isize));
            }
        }

        board
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..HEIGHT {
            f.write_str("|")?;
            for x in 0..WIDTH {
                match self.get(Coordinate::new(x as isize, (HEIGHT - 1 - y) as isize)) {
                    Cell::Empty => f.write_str(" "),
                    Cell::Filled(Player::Player1) => f.write_str("X"),
                    Cell::Filled(Player::Player2) => f.write_str("O"),
                }?;
            }
            f.write_str("|\n")?;
        }
        f.write_str("---\n")?;

        Ok(())
    }
}

impl Board {
    pub fn make_move(&mut self, mov: &BoardAction) -> Vec<MoveResult> {
        let mut results = Vec::new();
        match mov {
            BoardAction::DropStone(player, col) => {
                assert!(self.board[*col][HEIGHT - 1] == Cell::Empty);
                for y in 0..HEIGHT {
                    if self.board[*col][y] == Cell::Empty {
                        self.board[*col][y] = Cell::Filled(*player);
                        break;
                    }
                }
            }
            BoardAction::SwitchStone(a, b) => {
                let stone_a = self.get(*a);
                let stone_b = self.get(*b);

                self.set(stone_a, *b);
                self.set(stone_b, *a);
            }
        }

        loop {
            match self.get_board_terminal_status() {
                TerminalResult::None => {}
                TerminalResult::Win(player) => {
                    results.push(MoveResult::Winner(player));
                    return results;
                }
                TerminalResult::Draw => {
                    results.push(MoveResult::Draw);
                    return results;
                }
            }

            let (p1, ps1) = find_points(self, Player::Player1);
            let (p2, ps2) = find_points(self, Player::Player2);

            for _ in 0..p1 {
                results.push(MoveResult::Three(Player::Player1));
            }
            for _ in 0..p2 {
                results.push(MoveResult::Three(Player::Player2));
            }

            let mut total = HashSet::union(&ps1, &ps2).collect::<Vec<_>>();
            total.sort_by_key(|&c| (Reverse(c.y()), c.x()));

            // println!("{}", self);

            for coord in total {
                self.remove_stone(*coord);
            }

            if p1 == 0 && p2 == 0 {
                break;
            }
        }

        return results;
    }

    pub fn is_col_free(&self, col: usize) -> bool {
        self.board[col][HEIGHT - 1] == Cell::Empty
    }

    pub fn set(&mut self, cell: Cell, coord: Coordinate) {
        self.board[coord.x() as usize][coord.y() as usize] = cell;
    }

    pub fn get(&self, coord: Coordinate) -> Cell {
        if coord.is_contained((0, 0), (WIDTH as isize, HEIGHT as isize)) {
            self.board[coord.x() as usize][coord.y() as usize].clone()
        } else {
            Cell::Empty
        }
    }

    pub fn get_board_terminal_status(&self) -> TerminalResult {
        let mut player_1_four = 0;
        let mut player_2_four = 0;
        // Check horizontal lines starting left or right
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                match is_four_directional(self, Coordinate::new(x as isize, y as isize), (1, 0)) {
                    Some(Player::Player1) => player_1_four += 1,
                    Some(Player::Player2) => player_2_four += 1,
                    None => {}
                }
                match is_four_directional(self, Coordinate::new(x as isize, y as isize), (0, 1)) {
                    Some(Player::Player1) => player_1_four += 1,
                    Some(Player::Player2) => player_2_four += 1,
                    None => {}
                }
                match is_four_directional(self, Coordinate::new(x as isize, y as isize), (1, 1)) {
                    Some(Player::Player1) => player_1_four += 1,
                    Some(Player::Player2) => player_2_four += 1,
                    None => {}
                }
                match is_four_directional(self, Coordinate::new(x as isize, y as isize), (-1, 1)) {
                    Some(Player::Player1) => player_1_four += 1,
                    Some(Player::Player2) => player_2_four += 1,
                    None => {}
                }
            }
        }

        if player_1_four > 0 && player_2_four > 0 {
            TerminalResult::Draw
        } else if player_1_four == 0 && player_2_four == 0 {
            TerminalResult::None
        } else if player_1_four > 0 && player_2_four == 0 {
            TerminalResult::Win(Player::Player1)
        } else {
            TerminalResult::Win(Player::Player2)
        }
    }

    fn remove_stone(&mut self, mut coord: Coordinate) {
        self.board[coord.x() as usize][coord.y() as usize] = Cell::Empty;

        while coord.is_contained((0, 0), (WIDTH as isize, HEIGHT as isize)) {
            self.set(self.get(coord + (0, 1)), coord);
            coord = coord + (0, 1);
        }
    }
}

fn directional_stone_len(
    board: &Board,
    player: Player,
    coord: Coordinate,
    direction: (isize, isize),
) -> Vec<Coordinate> {
    let mut m = Vec::new();
    let mut current_coord = coord;

    while Cell::Filled(player) == board.get(current_coord) {
        m.push(current_coord);
        current_coord = current_coord + direction
    }
    m
}

fn is_four_directional(board: &Board, start: Coordinate, offset: (isize, isize)) -> Option<Player> {
    if let Cell::Filled(player) = board.get(start) {
        let forward = directional_stone_len(board, player, start, offset).len();
        let backward =
            directional_stone_len(board, player, start - offset, (-offset.0, -offset.1)).len();
        if forward == 4 && backward == 0 {
            return Some(player);
        }
    }

    return None;
}

fn find_points(board: &Board, player: Player) -> (usize, HashSet<Coordinate>) {
    let mut points = 0;
    let mut coords = HashSet::new();
    let mut up_set = HashSet::new();
    let mut up_right_set = HashSet::new();
    let mut right_set = HashSet::new();
    let mut down_right_set = HashSet::new();

    let mut check_direction =
        |coord: Coordinate, set: &mut HashSet<Coordinate>, direction: (isize, isize)| {
            if !set.contains(&coord) {
                let cells = directional_stone_len(board, player, coord, direction);
                if cells.len() >= 3 && cells.len() != 4 {
                    points += 1;
                    for coordinate in cells {
                        set.insert(coordinate);
                        coords.insert(coordinate);
                    }
                }
            }
        };

    // Horizontal
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let coord = Coordinate::new(x as isize, y as isize);
            check_direction(coord, &mut up_set, (0, 1));
            check_direction(coord, &mut up_right_set, (1, 1));
            check_direction(coord, &mut right_set, (1, 0));
            check_direction(coord, &mut down_right_set, (1, -1));
        }
    }

    (points, coords)
}

#[cfg(test)]
mod tests {
    use crate::{
        action::{BoardAction, Coordinate},
        board::MoveResult,
        player::Player,
    };

    use super::{Board, Cell};

    #[test]
    fn drop_stone() {
        let mut state = Board::default();
        let a = state.make_move(&BoardAction::DropStone(Player::Player1, 0));
        let b = state.make_move(&BoardAction::DropStone(Player::Player1, 0));
        let c = state.make_move(&BoardAction::DropStone(Player::Player1, 0));

        assert_eq!(a.len(), 0);
        assert_eq!(b.len(), 0);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0], MoveResult::Three(Player::Player1));
    }

    #[test]
    fn switch_stone() {
        let mut state = Board::default();
        assert_eq!(
            state
                .make_move(&BoardAction::DropStone(Player::Player1, 0))
                .len(),
            0
        );
        assert_eq!(
            state
                .make_move(&BoardAction::DropStone(Player::Player1, 1))
                .len(),
            0
        );
        assert_eq!(
            state
                .make_move(&BoardAction::DropStone(Player::Player2, 2))
                .len(),
            0
        );
        assert_eq!(
            state
                .make_move(&BoardAction::DropStone(Player::Player1, 3))
                .len(),
            0
        );
        let a = state.make_move(&BoardAction::SwitchStone(
            Coordinate::new(2, 0),
            Coordinate::new(3, 0),
        ));

        assert_eq!(a.len(), 1);
        assert_eq!(a[0], MoveResult::Three(Player::Player1));
    }

    #[test]
    fn multiple_three() {
        let board = [
            "XXO     ", "OOX     ", "XXO     ", "OOX     ", "XXO X   ", "OOX O   ", "XXO OXX ",
            "OOX XOOX",
        ];
        let mut state = Board::from(board);

        println!("{}", state);

        let results = state.make_move(&BoardAction::DropStone(Player::Player1, 3));

        println!("{}", state);

        // assert_eq!(results.len(), 1 + 9 + 1);
        assert_eq!(results[0], MoveResult::Three(Player::Player1));

        assert_eq!(results[1], MoveResult::Three(Player::Player1));
        assert_eq!(results[2], MoveResult::Three(Player::Player1));
        assert_eq!(results[3], MoveResult::Three(Player::Player1));
        assert_eq!(results[4], MoveResult::Three(Player::Player2));
        assert_eq!(results[5], MoveResult::Three(Player::Player2));
        assert_eq!(results[6], MoveResult::Three(Player::Player2));
        assert_eq!(results[7], MoveResult::Three(Player::Player2));
        assert_eq!(results[8], MoveResult::Three(Player::Player2));

        assert_eq!(results[9], MoveResult::Three(Player::Player1));

        let left = state
            .board
            .iter()
            .flat_map(|s| s.iter())
            .filter(|&&x| x != Cell::Empty)
            .count();

        assert_eq!(left, 4);
    }

    #[test]
    fn multiple_three_into_win() {
        let board = [
            "        ", "  OO    ", "  OO    ", "  XX    ", " XOO    ", " OXX    ", " XOO    ",
            "OOXX    ",
        ];
        let mut state = Board::from(board);

        println!("{}", state);

        let results = state.make_move(&BoardAction::DropStone(Player::Player1, 4));

        println!("{}", state);

        assert_eq!(results[0], MoveResult::Three(Player::Player1));
        assert_eq!(results[1], MoveResult::Winner(Player::Player2));
    }
}
