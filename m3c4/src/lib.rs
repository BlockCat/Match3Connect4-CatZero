use std::fmt::Debug;

use crate::board::{MoveResult, HEIGHT, WIDTH};
use action::{BoardAction, Coordinate};
use board::{Board, TerminalResult};
use catzero::Tensor;
use mcts::GameState;
use player::Player;

pub mod action;
pub mod alphazero;
pub mod board;
pub mod player;

#[derive(Default, Clone, Hash)]
pub struct BoardState {
    board: Board,
    player_1_points: usize,
    player_2_points: usize,
    current_player: Player,
    winner: TerminalResult,
}

impl Debug for BoardState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{}\n", self.board))?;
        f.write_str(&format!(
            "p1: {}, p2: {}'\n",
            self.player_1_points, self.player_2_points
        ))?;
        f.write_str(&format!("Turn: {:?}\n", self.current_player))?;
        f.write_str(&format!("Winner: {:?}\n", self.winner))?;
        Ok(())
    }
}

impl GameState for BoardState {
    type Move = BoardAction;
    type Player = Player;
    type MoveList = Vec<Self::Move>;

    fn current_player(&self) -> Self::Player {
        self.current_player.clone()
    }

    fn available_moves(&self) -> Self::MoveList {
        match self.board.get_board_terminal_status() {
            TerminalResult::None => {}
            TerminalResult::Win(_) => return Vec::new(),
            TerminalResult::Draw => return Vec::new(),
        }

        let mut actions: Self::MoveList = (0..board::WIDTH)
            .filter(|&col| self.board.is_col_free(col))
            .map(|col| BoardAction::DropStone(self.current_player(), col))
            .collect();

        let find_switch_actions = match self.current_player() {
            Player::Player1 => self.player_1_points > 0,
            Player::Player2 => self.player_2_points > 0,
        };

        if find_switch_actions {
            // Collect horizontal switches
            for x in 0..(board::WIDTH - 1) {
                for y in 0..board::HEIGHT {
                    let base_coord = Coordinate::new(x as isize, y as isize);
                    let next_coord = base_coord + (1, 0);
                    let next_cell = self.board.get(next_coord);
                    let add_action = match (next_cell, self.board.get(base_coord)) {
                        (board::Cell::Empty, board::Cell::Empty) => false,
                        (board::Cell::Empty, board::Cell::Filled(_)) => false,
                        (board::Cell::Filled(_), board::Cell::Empty) => false,
                        (
                            board::Cell::Filled(Player::Player1),
                            board::Cell::Filled(Player::Player1),
                        ) => false,
                        (
                            board::Cell::Filled(Player::Player2),
                            board::Cell::Filled(Player::Player2),
                        ) => false,
                        (
                            board::Cell::Filled(Player::Player1),
                            board::Cell::Filled(Player::Player2),
                        ) => true,
                        (
                            board::Cell::Filled(Player::Player2),
                            board::Cell::Filled(Player::Player1),
                        ) => true,
                    };
                    if add_action {
                        actions.push(BoardAction::SwitchStone(base_coord, next_coord));
                    }
                }
            }
            // Collect vertical switches
            for x in 0..(board::WIDTH - 1) {
                for y in 0..board::HEIGHT {
                    let base_coord = Coordinate::new(x as isize, y as isize);
                    let next_coord = base_coord + (0, 1);
                    let next_cell = self.board.get(next_coord);
                    let add_action = match (next_cell, self.board.get(base_coord)) {
                        (board::Cell::Empty, board::Cell::Empty) => false,
                        (board::Cell::Empty, board::Cell::Filled(_)) => false,
                        (board::Cell::Filled(_), board::Cell::Empty) => false,
                        (
                            board::Cell::Filled(Player::Player1),
                            board::Cell::Filled(Player::Player1),
                        ) => false,
                        (
                            board::Cell::Filled(Player::Player2),
                            board::Cell::Filled(Player::Player2),
                        ) => false,
                        (
                            board::Cell::Filled(Player::Player1),
                            board::Cell::Filled(Player::Player2),
                        ) => true,
                        (
                            board::Cell::Filled(Player::Player2),
                            board::Cell::Filled(Player::Player1),
                        ) => true,
                    };
                    if add_action {
                        actions.push(BoardAction::SwitchStone(base_coord, next_coord));
                    }
                }
            }
        }

        actions
    }

    fn make_move(&mut self, mov: &Self::Move) {
        if let BoardAction::SwitchStone(_, _) = mov {
            match self.current_player {
                Player::Player1 => self.player_1_points -= 1,
                Player::Player2 => self.player_2_points -= 1,
            }
        }

        let result = self.board.make_move(mov);
        let three_p1 = result
            .iter()
            .filter(|&x| x == &MoveResult::Three(Player::Player1))
            .count();
        let three_p2 = result
            .iter()
            .filter(|&x| x == &MoveResult::Three(Player::Player2))
            .count();

        self.player_1_points += three_p1;
        self.player_2_points += three_p2;

        self.current_player = self.current_player.next_player();

        self.winner = match result.last() {
            Some(MoveResult::Draw) => TerminalResult::Draw,
            Some(MoveResult::Winner(player)) => TerminalResult::Win(*player),
            _ => TerminalResult::None,
        };
    }

    fn get_winner(&self) -> Option<Self::Player> {
        match self.winner {
            TerminalResult::None => match self.board.get_board_terminal_status() {
                TerminalResult::None => None,
                TerminalResult::Win(player) => Some(player),
                TerminalResult::Draw => None,
            },
            TerminalResult::Win(player) => Some(player),
            TerminalResult::Draw => None,
        }
    }

    fn is_terminal(&self) -> bool {
        self.available_moves().is_empty()
    }
}

// Input: 8 x 8 planes
// -- History --
// 1 Binary Plane for X
// 1 Binary Plane for Y
// -- Other   --
// 1 Real Plane for points P1
// 1 Real Plane for points P2

// Output: 8 x 8 planes
// 1 Binary Plane for columns
// 1 Binary Plane for switch right
// 1 Binary Plane for switch up

fn tensor_to_tensorflow(tensor: Tensor<u8>) -> tensorflow::Tensor<f32> {
    let flattened = tensor
        .iter()
        .flat_map(|x| x.iter().flatten().map(|x| *x as f32))
        .collect::<Vec<_>>();
    let tensor = tensorflow::Tensor::new(&[1, 4, 8, 8]);

    tensor
        .with_values(&flattened)
        .expect("Could not use tensor")
}

impl Into<Tensor<u8>> for BoardState {
    fn into(self) -> Tensor<u8> {
        let player = self.current_player();
        let next_player = player.next_player();

        let mut cross_plane = vec![vec![0u8; 8]; 8];
        let mut circle_plane = vec![vec![0u8; 8]; 8];

        for x in 0..WIDTH {
            for y in 0..HEIGHT {
                cross_plane[x][y] = match self.board.get(Coordinate::new(x as isize, y as isize)) {
                    board::Cell::Filled(p) if p == player => 1,
                    _ => 0,
                };

                circle_plane[x][y] = match self.board.get(Coordinate::new(x as isize, y as isize)) {
                    board::Cell::Filled(p) if p == next_player => 1,
                    _ => 0,
                };
            }
        }

        let real_p1_plane = vec![vec![self.player_1_points as u8; 8]; 8];
        let real_p2_plane = vec![vec![self.player_2_points as u8; 8]; 8];

        vec![cross_plane, circle_plane, real_p1_plane, real_p2_plane]
    }
}

impl Into<tensorflow::Tensor<f32>> for BoardState {
    fn into(self) -> tensorflow::Tensor<f32> {
        tensor_to_tensorflow(self.into())
    }
}
