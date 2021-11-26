use m3c4::{action::BoardAction, player::Player, BoardState};
use mcts::{
    transposition_table::ApproxTable, tree_policy::UCTPolicy, Evaluator, GameState, MCTSManager,
    MCTS,
};
use rand::prelude::SliceRandom;

fn main() {
    println!("Starting program...");
    let mut state = BoardState::default();
    println!("Created initial state...");
    let exploration = 1.4;

    while !state.is_terminal() {
        let mut manager = MCTSManager::new(
            state.clone(),
            MyMCTS,
            RandomEvaluator,
            UCTPolicy::new(exploration),
            ApproxTable::new(1024),
        );
        println!("Created MCTS manager...");

        manager.playout_n_parallel(5000, 15);

        if let Some(best) = manager.best_move() {
            println!("Best move: {:?}", best);
            state.make_move(&best);
        }

        println!("{:?}", state);
    }
}

#[derive(Debug, Clone)]
enum StateEval {
    Win(Player),
    Draw,
}

struct MyMCTS;

impl MCTS for MyMCTS {
    type State = BoardState;
    type Eval = RandomEvaluator;
    type TreePolicy = UCTPolicy<()>;
    type NodeData = ();
    type TranspositionTable = ApproxTable<Self>;
    type ExtraThreadData = ();

    fn cycle_behaviour(&self) -> mcts::CycleBehaviour<Self> {
        mcts::CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}

struct RandomEvaluator;

impl Evaluator<MyMCTS> for RandomEvaluator {
    type StateEvaluation = StateEval;

    fn evaluate_new_state(
        &self,
        state: &BoardState,
        moves: &Vec<BoardAction>,
        _: Option<mcts::SearchHandle<MyMCTS>>,
    ) -> (Vec<mcts::MoveEvaluation<MyMCTS>>, Self::StateEvaluation) {
        let evals = moves.iter().map(|_| ()).collect();
        let mut rng = rand::thread_rng();
        let mut state = state.clone();

        while !state.is_terminal() {
            let moves = state.available_moves();
            let chosen = moves.choose(&mut rng).expect("Could not choose action");
            state.make_move(chosen);
        }

        let random_play_result = match state.get_winner() {
            Some(winner) => StateEval::Win(winner),
            None => StateEval::Draw,
        };

        (evals, random_play_result)
    }

    fn evaluate_existing_state(
        &self,
        _: &BoardState,
        existing_evaln: &Self::StateEvaluation,
        _: mcts::SearchHandle<MyMCTS>,
    ) -> Self::StateEvaluation {
        existing_evaln.clone()
    }

    fn interpret_evaluation_for_player(
        &self,
        evaluation: &Self::StateEvaluation,
        player: &mcts::Player<MyMCTS>,
    ) -> f64 {
        match evaluation {
            StateEval::Win(winner) if player == winner => 1.0,
            StateEval::Win(_) => -1.0,
            StateEval::Draw => 0.0,
        }
    }
}
