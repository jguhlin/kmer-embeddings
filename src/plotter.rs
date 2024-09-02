use std::{collections::HashMap, time::{Duration, Instant}};

use ratatui::{
    prelude::*,
    crossterm::event::{self, Event, KeyCode, KeyEventKind},
    widgets::*,
};

use crate::generator::KmerGenerator;

pub fn plot(kmer_size: u8) -> std::io::Result<()> {
    let mut generator = KmerGenerator::new();
        generator.set_dna();
        generator.set_k(kmer_size.into());
        generator.set_threads(8);

    let rx = generator.start();

    let mut terminal = ratatui::init();
    terminal.clear();

    let mut data = HashMap::new();

    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now();

    loop {
        if rx.len() > 0 {
            let get = rx.len();
            let mut iter = 0;
            while let Ok(result) = rx.recv() {
                data.entry(result.2.to_string()).and_modify(|e| *e += 1).or_insert(1);
        
                iter += 1;
                if iter >= get {
                    break;
                }
            }
        }

        terminal.draw(|frame| {

            let [top_area, main_area] = Layout::vertical([
                Constraint::Length(1),
                Constraint::Min(0),
            ]).areas(frame.area());

            frame.render_widget(
                Paragraph::new(format!("Press 'q' to quit - Process {} kmer pairs", data.values().sum::<u64>()))
                    .alignment(Alignment::Center),
                top_area
            );
            
            let mut bar_chart = plotter(&data);

            // Calculate the width of the bars for the chart
            let total_width = main_area.width as f64;
            let bar_width = (total_width / data.len() as f64).floor() as u16;
            let bar_width = bar_width.saturating_sub(2);
            let bar_width = bar_width.max(1);

            bar_chart = bar_chart.bar_width(bar_width);


            frame.render_widget(
                bar_chart,                
                main_area
            );
        })?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.code == KeyCode::Char('q') {
                        break;
                    }
                }
            }
            if last_tick.elapsed() >= tick_rate {
                last_tick = std::time::Instant::now();
            }
    }
    ratatui::restore();

    println!("{:#?}", data);

    // Get custom max value and add 10% to it
    let total = data.values().sum::<u64>();

    let mut bar_chart_data = data.iter().map(|(k, v)| {
        (k.as_str(), k.parse::<u64>().unwrap(), ((*v as f64 / total as f64) * 100 as f64) as u64)
    }).collect::<Vec<_>>();

    bar_chart_data.sort_by(|a, b| a.1.cmp(&b.1));

    // Drop second value
    let bar_chart_data = bar_chart_data.into_iter().map(|(k, _, v)| {
        (k, v)
    }).collect::<Vec<_>>();

    println!("{:#?}", bar_chart_data);

    Ok(())
}

pub fn plotter<'a>(data: &'a HashMap<String, u64>) -> BarChart<'a> {
        // Create the datasets to fill the chart with

        // Get custom max value and add 10% to it
        let total = data.values().sum::<u64>();

        let mut bar_chart_data = data.iter().map(|(k, v)| {
            (k.as_str(), k.parse::<u64>().unwrap(), ((*v as f64 / total as f64) * 100 as f64) as u64)
        }).collect::<Vec<_>>();

        bar_chart_data.sort_by(|a, b| a.1.cmp(&b.1));

        // Drop second value
        let bar_chart_data = bar_chart_data.into_iter().map(|(k, _, v)| {
            (k, v)
        }).collect::<Vec<_>>();


        // Create the chart
        let chart = BarChart::default()
            .block(Block::bordered().title("Kmer Distances Sampled"))
            .bar_width(1)
            .bar_gap(1)
            .max(100)
            .bar_style(Style::new().blue())
            .value_style(Style::new().red().bold())
            .label_style(Style::new().white())
            .data(&bar_chart_data[..]);


        chart
    
}