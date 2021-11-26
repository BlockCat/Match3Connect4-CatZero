use std::ops::{Add, Sub};

use crate::player::Player;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Coordinate(isize, isize);

impl Coordinate {
    pub fn new(x: isize, y: isize) -> Self {
        Coordinate(x, y)
    }
    pub fn x(&self) -> isize {
        self.0
    }
    pub fn y(&self) -> isize {
        self.1
    }

    // a: Inclusive, b: Exclusive
    pub fn is_contained(&self, a: (isize, isize), b: (isize, isize)) -> bool {
        (a.0..b.0).contains(&self.0) && (a.1..b.1).contains(&self.1)
    }

    pub fn offset(&self, offset: (isize, isize), distance: isize) -> Self {
        Coordinate(self.0 + offset.0 * distance, self.1 + offset.0 * distance)
    }
}

impl Add<(isize, isize)> for Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: (isize, isize)) -> Self::Output {
        Coordinate(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Sub<(isize, isize)> for Coordinate {
    type Output = Coordinate;

    fn sub(self, rhs: (isize, isize)) -> Self::Output {
        Coordinate(self.0 - rhs.0, self.1 - rhs.1)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BoardAction {
    DropStone(Player, usize),
    SwitchStone(Coordinate, Coordinate),
}
