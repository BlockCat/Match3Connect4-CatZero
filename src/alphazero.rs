use crate::{action::BoardAction, player::Player, BoardState};
use catzero::{AlphaEvaluator, AlphaGame, TFModel};
use mcts::{
    transposition_table::ApproxTable, tree_policy::UCTPolicy, CycleBehaviour, GameState,
    MCTSManager, MCTS,
};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum StateEval {
    Winner(Player),
    Draw,
    Evaluation(Player, f32),
}

#[derive(Clone)]
pub struct MyMCTS {
    exploration_constant: f64,
    playouts: usize,
}

impl MyMCTS {
    pub fn create_manager(
        state: BoardState,
        exploration_constant: f64,
        playouts: usize,
        model: Arc<TFModel>,
    ) -> MCTSManager<MyMCTS> {
        let manager = MyMCTS {
            exploration_constant,
            playouts,
        };
        let eval = AlphaEvaluator::new(state.current_player(), model);
        let tree_policy = UCTPolicy::new(exploration_constant);
        MCTSManager::new(state, manager, eval, tree_policy, ApproxTable::new(1024))
    }
}

impl AlphaGame for MyMCTS {
    fn create_manager(
        state: BoardState,
        exploration_constant: f64,
        playouts: usize,
        model: Arc<TFModel>,
    ) -> MCTSManager<Self> {
        let manager = MyMCTS {
            exploration_constant,
            playouts,
        };
        let eval = AlphaEvaluator::new(state.current_player(), model);
        let tree_policy = UCTPolicy::new(exploration_constant);
        MCTSManager::new(state, manager, eval, tree_policy, ApproxTable::new(1024))
    }

    fn get_exploration(&self) -> f64 {
        self.exploration_constant
    }

    fn get_playouts(&self) -> usize {
        self.playouts
    }

    fn moves_to_evaluation(
        moves: &mcts::MoveList<Self>,
        policy: tensorflow::Tensor<f32>,
    ) -> Vec<mcts::MoveEvaluation<Self>> {
        let policy = policy.iter().map(|d| *d as f64).collect::<Vec<_>>();
        let policy = tensorflow::Tensor::new(&[1, 3, 8, 8])
            .with_values(&policy)
            .expect("Could not reshape");

        moves
            .iter()
            .map(|mov| match mov {
                BoardAction::DropStone(_, col) => policy.get(&[0, 0, *col as u64, 0]),
                BoardAction::SwitchStone(a, b) if a.x() == b.x() => {
                    policy.get(&[0, 1, a.x() as u64, a.y().min(b.y()) as u64])
                }
                BoardAction::SwitchStone(a, b) if a.y() == b.y() => {
                    policy.get(&[0, 2, a.x().min(b.x()) as u64, a.y() as u64])
                }
                _ => unreachable!(),
            })
            .collect()
    }

    fn moves_to_tensorflow(moves: Vec<&mcts::MoveInfo<Self>>) -> tensorflow::Tensor<f32> {
        let mut tensor = tensorflow::Tensor::new(&[1, 3, 8, 8]);
        let parent_visits: u64 = moves.iter().map(|&x| x.visits()).sum();

        if parent_visits == 0 {
            panic!("Parent visits were 0");
        }

        let parent_visits = parent_visits as f32;

        for m in moves {
            let visit = m.visits() as f32;
            let probability = visit / parent_visits;
            let indeces: [u64; 4] = match m.get_move() {
                BoardAction::DropStone(_, col) => [0, 0, *col as u64, 0],
                BoardAction::SwitchStone(a, b) if a.x() == b.x() => {
                    [0, 1, a.x() as u64, a.y().min(b.y()) as u64]
                }
                BoardAction::SwitchStone(a, b) if a.y() == b.y() => {
                    [0, 2, a.x().min(b.x()) as u64, a.y() as u64]
                }
                _ => unreachable!(),
            };

            tensor.set(&indeces, probability);
        }

        tensor
    }
}

impl MCTS for MyMCTS {
    type State = BoardState;
    type Eval = AlphaEvaluator<Self>;
    type TreePolicy = UCTPolicy<f64>;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}
