use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::fmt;

use rand::prelude::*;

use clap::{Parser, Subcommand};
use mcts::*;
use mcts::tree_policy::*;
use mcts::transposition_table::*;

const LOW_CARD: usize = 3;
const HIGH_CARD: usize = 35;
const DISCARDED_CARDS: usize = 9;
const NUM_CARDS: usize = HIGH_CARD - LOW_CARD + 1;

#[derive(Clone, Debug, PartialEq, Hash)]
struct NoThanksGame {
    active_tokens: usize,
    active_card: Option<usize>,
    active_player: usize,
    cards_taken: usize,
    player_tokens: Vec<usize>,
    card_owners: [Option<usize>; NUM_CARDS],
}

impl fmt::Display for NoThanksGame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.active_card {
            Some(card) => {
                write!(f, "{}", card + LOW_CARD)?;
                if self.active_tokens > 0 {
                    write!(f, " ({} tokens)", self.active_tokens)?
                }
                write!(f, "\n")?;
            },
            None => (),
        }
        for (i, tokens) in self.player_tokens.iter().enumerate() {
            write!(f, "{}Player {} ({} tokens): ",
                    if self.active_player == i { "*" } else { "" }, i, tokens)?;
            for (card, owner) in self.card_owners.iter().enumerate() {
                match *owner {
                    None => (),
                    Some(owner) => {
                        if owner == i {
                            write!(f, "{}, ", card + LOW_CARD)?;
                        }
                    },
                }
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Move {
    Pass,
    Take,
    NextCard(usize),
}

enum Player {
    Random,
    Player(usize),
}

impl NoThanksGame {
    fn new(num_players: usize) -> Self {
        let starting_tokens = match num_players {
            3 | 4 | 5 => 11,
            6 => 9,
            7 => 7,
            _ => panic!("Invalid number of players"),
        };
        NoThanksGame {
            active_tokens: 0,
            active_card: None,
            active_player: 0,
            cards_taken: 0,
            player_tokens: vec![starting_tokens; num_players],
            card_owners: [None; NUM_CARDS],
        }
    }

    fn is_terminal(&self) -> bool {
        self.cards_taken >= NUM_CARDS - DISCARDED_CARDS
    }

    fn compute_scores(&self) -> Vec<i64> {
        let mut last_owner = None;
        let mut scores: Vec<i64> = self.player_tokens.iter().map(|t| -(*t as i64)).collect();
        for (i, owner) in self.card_owners.iter().enumerate() {
            if *owner == last_owner {
                continue;
            }
            last_owner = *owner;
            match *owner {
                None => (),
                Some(owner) => {
                    scores[owner] += (i + LOW_CARD) as i64;
                },
            }
        }
        scores
    }

}

impl GameState for NoThanksGame {
    type Move = Move;
    type Player = Player;
    type MoveList = Vec<Move>;

    fn current_player(&self) -> Self::Player {
        match self.active_card {
            None => Player::Random,
            Some(_) => Player::Player(self.active_player),
        }
    }

    fn available_moves(&self) -> Self::MoveList {
        match self.active_card {
            None => {
                if !self.is_terminal() {
                    self.card_owners
                        .iter()
                        .enumerate()
                        .filter_map(|(i, owner)| match owner {
                            None => Some(Move::NextCard(i)),
                            Some(_) => None,
                        })
                        .collect()
                } else {
                    vec![]
                }
            },
            Some(_) => {
                if self.player_tokens[self.active_player] > 0 {
                    vec![Move::Pass, Move::Take]
                } else {
                    vec![Move::Take]
                }
            },
        }
    }

    fn make_move(&mut self, mov: &Self::Move) {
        match *mov {
            Move::NextCard(card) => {
                self.active_card = Some(card);
            },
            Move::Pass => {
                self.active_tokens += 1;
                self.player_tokens[self.active_player] -= 1;
                self.active_player = (self.active_player + 1) % self.player_tokens.len();
            },
            Move::Take => {
                self.player_tokens[self.active_player] += self.active_tokens;
                self.active_tokens = 0;
                self.card_owners[self.active_card.unwrap()] = Some(self.active_player);
                self.active_card = None;
                self.cards_taken += 1;
            },
        }
    }
}

impl TranspositionHash for NoThanksGame {
    fn hash(&self) -> u64 {
        let mut s = DefaultHasher::new();
        Hash::hash(&self, &mut s);
        s.finish()
    }
}

struct MyEvaluator;

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = Vec<i64>; // positive: win by that margin (relative to next player); negative: loss by that margin (relative to first player)

    fn evaluate_new_state(&self, state: &NoThanksGame, moves: &Vec<Move>, _: Option<SearchHandle<MyMCTS>>) -> (Vec<()>, Self::StateEvaluation) {
        (vec![(); moves.len()], state.compute_scores())
    }

    fn interpret_evaluation_for_player(&self, evaln: &Self::StateEvaluation, player: &Player) -> i64 {
        match player {
            Player::Random => 0,
            Player::Player(i) => -evaln[*i],
        }
    }

    fn evaluate_existing_state(&self, _state: &NoThanksGame, evaln: &Self::StateEvaluation, _: SearchHandle<MyMCTS>) -> Self::StateEvaluation {
        evaln.clone()
    }
}

#[derive(Default)]
struct MyMCTS;

impl MCTS for MyMCTS {
    type State = NoThanksGame;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ExtraThreadData = ();
    type TreePolicy = UCTPolicy;
    type TranspositionTable = ApproxTable<Self>;

    fn cycle_behaviour(&self) -> CycleBehaviour<Self> {
        CycleBehaviour::UseCurrentEvalWhenCycleDetected
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    SelfPlay {
        #[arg(short, long)]
        players: usize,
    },
    WithHumans {
        #[arg(short, long)]
        players: usize,
        #[arg(short, long)]
        which_player: usize,
    },
}

fn self_play(players: usize) {
    let mut game = NoThanksGame::new(players);
    let mut rng = rand::thread_rng();
    while !game.is_terminal() {
        match game.current_player() {
            Player::Random => {
                // Choose a random move
                game.make_move(&game.available_moves().iter().choose(&mut rng).unwrap());
                println!("{}", game);
            },
            Player::Player(i) => {
                let mut mcts = MCTSManager::new(game.clone(), MyMCTS, MyEvaluator, UCTPolicy::new(0.5), ApproxTable::new(1024));
                mcts.playout_n_parallel(1000000, 8);
                let best_move = mcts.best_move().unwrap();
                match best_move {
                    Move::NextCard(_) => panic!("impossible"),
                    Move::Pass => print!("{} passes, ", i),
                    Move::Take => {
                        println!("{} takes at {} tokens\n", i, game.active_tokens);
                    }
                }
                game.make_move(&best_move);
            },
        }
    }

    println!("{}", game);
    println!("{:?}", game.compute_scores());
}

fn get_input_number() -> usize {
    loop {
        let mut buffer = String::new();
        match std::io::stdin().read_line(&mut buffer) {
            Ok(_) => {
                match buffer.trim().parse::<usize>() {
                    Ok(n) => break n,
                    Err(_) => println!("Enter a number:"),
                }
            },
            Err(_) => println!("Enter a number:"),
        }
    }
}

fn with_humans(players: usize, which_player: usize) {
    let mut game = NoThanksGame::new(players);
    let mut game_at_last_card = game.clone();
    while !game.is_terminal() {
        match game.current_player() {
            Player::Random => {
                println!("Enter next card:");
                let card = get_input_number();
                game.make_move(&Move::NextCard(card - 3));
                game_at_last_card = game.clone();
                println!("{}", game);
            }
            Player::Player(i) => {
                let found_best_move =
                    if i == which_player {
                        let mut mcts = MCTSManager::new(game.clone(), MyMCTS, MyEvaluator, UCTPolicy::new(0.5), ApproxTable::new(1024));
                        mcts.playout_n_parallel(1000000, 8);
                        let best_move = mcts.best_move().unwrap();
                        match best_move {
                            Move::NextCard(_) => panic!("impossible"),
                            Move::Pass => {
                                game.make_move(&Move::Pass);
                                false
                            },
                            Move::Take => {
                                println!("Take at {} tokens\n", game.active_tokens);
                                true
                            }
                        }
                    } else {
                        if game.player_tokens[game.active_player] == 0 {
                            game.make_move(&Move::Take);
                            println!("Never take\n");
                            true
                        } else {
                            game.make_move(&Move::Pass);
                            false
                        }
                    };
                if found_best_move {
                    println!("When was card taken:");
                    let tokens = get_input_number();
                    while game_at_last_card.active_tokens < tokens {
                        game_at_last_card.make_move(&Move::Pass);
                    }
                    game_at_last_card.make_move(&Move::Take);
                    game = game_at_last_card.clone();
                }
            },
        }
    }

    println!("{}", game);
    println!("{:?}", game.compute_scores());
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Command::SelfPlay { players } => self_play(players),
        Command::WithHumans { players, which_player } => with_humans(players, which_player),
    }
}
