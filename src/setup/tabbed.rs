//! One transaction-shaped setup UI. Choices stay in memory until final Accept.

use std::io;

use anyhow::{Context, Result, bail};
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::Line,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Tabs, Wrap},
};

use super::wizard::MenuItem;

pub struct Page {
    pub title: String,
    pub help: String,
    pub items: Vec<MenuItem>,
    pub preferred: usize,
}

impl Page {
    pub fn new(
        title: impl Into<String>,
        help: impl Into<String>,
        items: Vec<MenuItem>,
        preferred: usize,
    ) -> Self {
        Self {
            title: title.into(),
            help: help.into(),
            items,
            preferred,
        }
    }
}

#[derive(Debug)]
struct State {
    tab: usize,
    hover: usize,
    choices: Vec<Option<usize>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Change {
    Redraw,
    Finish,
    Cancel,
}

impl State {
    fn new(tabs: usize) -> Self {
        Self {
            tab: 0,
            hover: 0,
            choices: vec![None; tabs],
        }
    }

    fn sync(&mut self, page: &Page) -> Result<()> {
        if page.items.is_empty() || !page.items.iter().any(|item| item.enabled) {
            bail!("{} has no available choices", page.title);
        }
        self.hover = self.choices[self.tab]
            .filter(|index| page.items.get(*index).is_some_and(|item| item.enabled))
            .or_else(|| {
                page.items
                    .get(page.preferred)
                    .filter(|item| item.enabled)
                    .map(|_| page.preferred)
            })
            .unwrap_or_else(|| {
                page.items
                    .iter()
                    .position(|item| item.enabled)
                    .unwrap_or_default()
            });
        Ok(())
    }

    fn handle(&mut self, key: KeyCode, page: &Page) -> Change {
        match key {
            KeyCode::Up | KeyCode::Char('k') => self.move_hover(&page.items, -1),
            KeyCode::Down | KeyCode::Char('j') => self.move_hover(&page.items, 1),
            KeyCode::Home => {
                self.hover = page
                    .items
                    .iter()
                    .position(|item| item.enabled)
                    .unwrap_or(self.hover)
            }
            KeyCode::End => {
                self.hover = page
                    .items
                    .iter()
                    .rposition(|item| item.enabled)
                    .unwrap_or(self.hover)
            }
            KeyCode::Char(' ') if page.items[self.hover].enabled => {
                let changed = self.choices[self.tab] != Some(self.hover);
                self.choices[self.tab] = Some(self.hover);
                if changed {
                    self.choices
                        .iter_mut()
                        .skip(self.tab + 1)
                        .for_each(|choice| *choice = None);
                }
            }
            KeyCode::Left if self.tab > 0 => {
                self.tab -= 1;
            }
            KeyCode::Right
                if self.choices[self.tab].is_some() && self.tab + 1 < self.choices.len() =>
            {
                self.tab += 1;
            }
            KeyCode::Enter if self.choices[self.tab].is_some() => {
                if self.tab + 1 == self.choices.len() {
                    return Change::Finish;
                }
                self.tab += 1;
            }
            KeyCode::Esc | KeyCode::Char('q') => return Change::Cancel,
            _ => {}
        }
        Change::Redraw
    }

    fn move_hover(&mut self, items: &[MenuItem], direction: isize) {
        let mut next = self.hover;
        loop {
            next = (next as isize + direction).rem_euclid(items.len() as isize) as usize;
            if items[next].enabled {
                self.hover = next;
                return;
            }
        }
    }
}

pub fn run(
    tabs: &[&str],
    page: impl Fn(usize, &[Option<usize>]) -> Result<Page>,
) -> Result<Option<Vec<usize>>> {
    if tabs.is_empty() {
        bail!("setup has no pages");
    }
    let _session = TerminalSession::enter()?;
    let mut terminal =
        Terminal::new(CrosstermBackend::new(io::stdout())).context("open setup terminal")?;
    let mut state = State::new(tabs.len());
    let mut last_tab = None;
    loop {
        let current = page(state.tab, &state.choices)?;
        if last_tab != Some(state.tab) {
            state.sync(&current)?;
            last_tab = Some(state.tab);
        }
        terminal.draw(|frame| render(frame, tabs, &current, &state))?;
        let Event::Key(key) = event::read().context("read setup key")? else {
            continue;
        };
        if !matches!(key.kind, KeyEventKind::Press | KeyEventKind::Repeat)
            || !matches!(key.modifiers, KeyModifiers::NONE | KeyModifiers::SHIFT)
        {
            continue;
        }
        match state.handle(key.code, &current) {
            Change::Finish => {
                return Ok(Some(
                    state.choices.into_iter().map(Option::unwrap).collect(),
                ));
            }
            Change::Cancel => return Ok(None),
            Change::Redraw => {}
        }
    }
}

fn render(frame: &mut ratatui::Frame, tabs: &[&str], page: &Page, state: &State) {
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Length(7),
            Constraint::Min(3),
            Constraint::Length(4),
            Constraint::Length(1),
        ])
        .split(frame.area());
    let labels = tabs
        .iter()
        .enumerate()
        .map(|(index, title)| {
            if state.choices[index].is_some() {
                format!("✓ {title}")
            } else {
                (*title).to_owned()
            }
        })
        .collect::<Vec<_>>();
    let tab_bar = Tabs::new(labels)
        .select(state.tab)
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .divider(" │ ")
        .block(Block::default().borders(Borders::BOTTOM));
    frame.render_widget(tab_bar, rows[0]);
    frame.render_widget(
        Line::styled(&page.title, Style::default().add_modifier(Modifier::BOLD)),
        rows[1],
    );
    frame.render_widget(
        Paragraph::new(page.help.as_str()).wrap(Wrap { trim: true }),
        rows[2],
    );

    let items = page
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let mark = if state.choices[state.tab] == Some(index) {
                "x"
            } else {
                " "
            };
            let style = if item.enabled {
                Style::default()
            } else {
                Style::default().fg(Color::DarkGray)
            };
            ListItem::new(format!("[{mark}] {}", item.label)).style(style)
        })
        .collect::<Vec<_>>();
    let list = List::new(items)
        .highlight_symbol("› ")
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .block(Block::default().borders(Borders::TOP | Borders::BOTTOM));
    let mut list_state = ListState::default().with_selected(Some(state.hover));
    frame.render_stateful_widget(list, rows[3], &mut list_state);
    frame.render_widget(
        Paragraph::new(page.items[state.hover].detail.as_str())
            .block(Block::default().title("Details").borders(Borders::BOTTOM))
            .wrap(Wrap { trim: true }),
        rows[4],
    );
    frame.render_widget(
        Line::styled(
            "↑↓ move  Space select  ←→ tabs  Enter continue  Esc cancel",
            Style::default().fg(Color::DarkGray),
        ),
        rows[5],
    );
}

struct TerminalSession;

impl TerminalSession {
    fn enter() -> Result<Self> {
        terminal::enable_raw_mode().context("enable setup raw mode")?;
        if let Err(error) = execute!(io::stdout(), EnterAlternateScreen, cursor::Hide) {
            let _ = terminal::disable_raw_mode();
            return Err(error).context("open setup screen");
        }
        Ok(Self)
    }
}

impl Drop for TerminalSession {
    fn drop(&mut self) {
        let _ = execute!(io::stdout(), cursor::Show, LeaveAlternateScreen);
        let _ = terminal::disable_raw_mode();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn page() -> Page {
        Page::new(
            "Runtime",
            "Choose a provider",
            vec![
                MenuItem::available("CPU", "Packaged provider"),
                MenuItem::unavailable("CUDA", "Missing provider"),
                MenuItem::available("OpenVINO", "Intel NPU"),
            ],
            0,
        )
    }

    #[test]
    fn space_selects_enter_advances_and_revising_clears_later_choices() {
        let mut state = State::new(3);
        let page = page();
        state.sync(&page).unwrap();
        assert_eq!(state.handle(KeyCode::Enter, &page), Change::Redraw);
        assert_eq!(state.tab, 0);
        state.handle(KeyCode::Down, &page);
        assert_eq!(state.hover, 2);
        state.handle(KeyCode::Char(' '), &page);
        assert_eq!(state.choices[0], Some(2));
        state.handle(KeyCode::Enter, &page);
        assert_eq!(state.tab, 1);
        state.choices[1] = Some(0);
        state.handle(KeyCode::Left, &page);
        state.sync(&page).unwrap();
        state.handle(KeyCode::Up, &page);
        state.handle(KeyCode::Char(' '), &page);
        assert_eq!(state.choices, [Some(0), None, None]);
    }

    #[test]
    fn ratatui_frame_shows_tabs_options_and_instructions() {
        let mut terminal = Terminal::new(ratatui::backend::TestBackend::new(80, 24)).unwrap();
        let mut state = State::new(3);
        let page = page();
        state.sync(&page).unwrap();
        terminal
            .draw(|frame| render(frame, &["Runtime", "Model", "Accept"], &page, &state))
            .unwrap();
        let text = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|cell| cell.symbol())
            .collect::<String>();
        assert!(text.contains("Runtime"));
        assert!(text.contains("[ ] CPU"));
        assert!(text.contains("[ ] OpenVINO"));
        assert!(text.contains("Space select"));
    }

    #[test]
    fn accept_requires_space_and_left_restores_the_previous_choice() {
        let accept_page = Page::new(
            "Accept",
            "Apply only after confirmation",
            vec![MenuItem::available("Apply", "Save settings")],
            0,
        );
        let mut state = State {
            tab: 1,
            hover: 0,
            choices: vec![Some(2), None],
        };
        state.sync(&accept_page).unwrap();
        assert_eq!(state.handle(KeyCode::Enter, &accept_page), Change::Redraw);
        state.handle(KeyCode::Left, &accept_page);
        assert_eq!(state.tab, 0);
        state.sync(&page()).unwrap();
        assert_eq!(state.hover, 2);
        state.handle(KeyCode::Right, &page());
        state.sync(&accept_page).unwrap();
        state.handle(KeyCode::Char(' '), &accept_page);
        assert_eq!(state.handle(KeyCode::Enter, &accept_page), Change::Finish);
    }
}
