use m3c4::BoardState;

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

    let python_model = if start == 0 {
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
    let state = BoardState::default();

    let model = python_model
        .to_tf_model(0)
        .expect("Could not create tensor model");

    let res = model.evaluate(state.into()).expect("Could not ");

    println!("{:?}", res);
}
