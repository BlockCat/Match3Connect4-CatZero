use catzero::{AlphaGame, TFModel, Tensor, TrainingData};
use m3c4::{alphazero::MyMCTS, player::Player, BoardState};
use mcts::GameState;
use rand::prelude::SliceRandom;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

const EXPLORATION: f64 = 1.45;
const GAMES_TO_PLAY: usize = 25;
const PLAYOUTS: usize = 500;

const EPISODES: usize = 80;
const BATCH_SIZE: u32 = 20;
const EPOCHS: u32 = 100;

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

fn main() {
    let mut pyenv = catzero::PyEnv::new();
    let python = pyenv.python();

    let start = 0;

    let mut python_model = if start == 0 {
        catzero::CatZeroModel::new(
            &python,
            (4, 8, 8),
            (3, 8, 8),
            0.001,
            1.0,
            10,
            String::from("data/models/graph"),
        )
        .expect("Could not create new model")
    } else {
        catzero::CatZeroModel::load(&python, "data/models/graph", start, (1, 3, 3))
            .expect("Could not load model")
    };

    for episode in start..EPISODES {
        let model = python_model
            .to_tf_model(episode)
            .expect("Could not create tensor model");
        let model = Arc::new(model);

        // let mut results = Vec::new();
        println!("Starting episode: {}", episode);

        let results = (0..GAMES_TO_PLAY)
            .into_par_iter()
            .map(|i| {
                println!("Starting a game: {}", i);
                let res = play_a_game(model.clone());
                println!("Played a game: {}", i);
                res
            })
            .collect::<Vec<_>>();

        let inputs: Vec<Tensor<u8>> = results
            .iter()
            .flat_map(|result| result.histories.iter())
            .map(|(state, _)| state.clone().into())
            .collect();

        println!(
            "Collected: {} states in {} games, during episode {}",
            inputs.len(),
            GAMES_TO_PLAY,
            episode
        );

        let output_policy: Vec<Tensor<f32>> = results
            .iter()
            .flat_map(|result| result.histories.iter())
            .map(|(_, tensor)| {
                tensor
                    .chunks(8 * 8)
                    .map(|s| s.chunks(8).map(|d| d.to_vec()).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect();

        let output_value: Vec<f32> = results
            .iter()
            .flat_map(|result| {
                result.histories.iter().map(move |(s, _)| {
                    match (s.current_player(), &result.winner) {
                        (Player::Player1, Some(Player::Player1)) => 1.0,
                        (Player::Player1, Some(Player::Player2)) => -1.0,
                        (Player::Player2, Some(Player::Player1)) => -1.0,
                        (Player::Player2, Some(Player::Player2)) => 1.0,
                        (_, None) => 0.0,
                    }
                })
                // result.histories.iter().map(move |_| reward)
            })
            .collect();

        assert!(inputs.len() == output_policy.len());
        assert!(inputs.len() == output_value.len());

        let data = TrainingData {
            inputs,
            output_policy,
            output_value,
        };

        data.print(0..data.len().min(10));

        if let Err(e) = data.save(&format!("data/{}.games", episode)) {
            println!("Did not save game data: {}", e);
        }

        std::iter::repeat_with(|| python_model.learn(&data, BATCH_SIZE, EPOCHS))
            .take(10)
            .find(|a| match a {
                Ok(_) => {
                    println!("Learned an episode");
                    true
                }
                Err(_) => {
                    println!("Failed learning");
                    false
                }
            })
            .expect("Could not learn after 10 retries")
            .unwrap();
    }
}

// play a game and a list of states
fn play_a_game(model: Arc<TFModel>) -> GameResult {
    let mut rng = rand::thread_rng();
    let mut state = BoardState::default();

    let mut histories = Vec::new();

    while !state.is_terminal() {
        let mut mcts_manager =
            MyMCTS::create_manager(state.clone(), EXPLORATION, PLAYOUTS, model.clone());

        mcts_manager.playout_n(PLAYOUTS);

        let root_node = mcts_manager.tree().root_node();
        let moves = root_node.moves().collect::<Vec<_>>();

        histories.push((state.clone(), MyMCTS::moves_to_tensorflow(moves.clone())));

        let weighted_action = moves
            .choose_weighted(&mut rng, |i| i.visits())
            .expect("Could not get a random action");

        state.make_move(weighted_action.get_move());
    }

    println!("final: {:?}", state);

    GameResult::new(state.get_winner(), histories)
}

struct GameResult {
    histories: Vec<(BoardState, tensorflow::Tensor<f32>)>,
    winner: Option<Player>,
}

impl GameResult {
    pub fn new(
        winner: Option<Player>,
        histories: Vec<(BoardState, tensorflow::Tensor<f32>)>,
    ) -> GameResult {
        Self { histories, winner }
    }
}
