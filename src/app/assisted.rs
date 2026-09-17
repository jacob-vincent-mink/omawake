//! Optional Omaspeak integration. Only its file-output API is used for generation;
//! playback is a separate, explicit pronunciation-review action.
use super::*;
use crate::enrollment::{
    Cancellation, SampleSet,
    artifact::{Dataset, GeneratedRecording, LabeledRecording},
};
use crate::setup::wizard::{MenuItem, select};
use std::os::unix::{fs::OpenOptionsExt, process::CommandExt};

pub(super) fn executable(name: &str) -> Option<PathBuf> {
    let mut dirs: Vec<_> =
        std::env::split_paths(&std::env::var_os("PATH").unwrap_or_default()).collect();
    if name == "omaspeak"
        && let Some(home) = std::env::var_os("HOME")
    {
        dirs.push(PathBuf::from(home).join(".local/bin"));
    }
    dirs.into_iter().map(|dir| dir.join(name)).find(|path| {
        fs::metadata(path).is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    })
}

/// Installation is a user-selected step, never a side effect of discovery.
pub(super) fn prepare() -> Result<bool> {
    prepare_with(executable, choose, |curl, tar| {
        let home = PathBuf::from(std::env::var_os("HOME").context("HOME is not set")?);
        let paths = AppPaths::discover();
        let scratch = SampleSet::create(&paths)?;
        install_release(curl, tar, &home, scratch.directory(), &Cancellation::new()?)
    })
}

fn prepare_with(
    discover: impl Fn(&str) -> Option<PathBuf>,
    mut ask: impl FnMut(&str, &str, &[MenuItem]) -> Result<usize>,
    install: impl FnOnce(&Path, &Path) -> Result<()>,
) -> Result<bool> {
    if discover("omaspeak").is_none() {
        let installer = discover("curl").zip(discover("tar"));
        let install_item = match &installer {
            Some(_) => MenuItem::available(
                "Install Omaspeak",
                "Download verified Omaspeak 0.0.1 to ~/.local; no sudo or service changes",
            ),
            None => MenuItem::unavailable(
                "Install Omaspeak",
                "Install from https://github.com/jacob-vincent-mink/omaspeak/releases, then return here",
            ),
        };
        match ask(
            "Omaspeak is required",
            "Generated examples are optional. A configured local TTS model is also needed.",
            &[
                MenuItem::available(
                    "Continue with human recordings only",
                    "No software is installed",
                ),
                install_item,
            ],
        )? {
            0 => return Ok(false),
            1 => {
                let (curl, tar) =
                    installer.context("curl and tar are required for installation")?;
                install(&curl, &tar)?;
            }
            _ => bail!("onboarding cancelled; configuration unchanged"),
        }
    }
    anyhow::ensure!(
        discover("omaspeak").is_some(),
        "Omaspeak is still missing from PATH"
    );
    Ok(true)
}

fn release_asset(arch: &str) -> Result<(&'static str, &'static str)> {
    match arch {
        "x86_64" => Ok((
            "omaspeak-0.0.1-linux-x86_64",
            "9e318960fb15fdf955efbb8dda9bc8eb2b9d0932a9acd9e31b85bf3492ca78ea",
        )),
        "aarch64" => Ok((
            "omaspeak-0.0.1-linux-aarch64",
            "8a0d7728d0b6d3f447ab7a616389fc8c54a0617f14922b33593ca60b425deb8d",
        )),
        _ => bail!("no bundled Omaspeak installer for this architecture; install it manually"),
    }
}
fn install_release(
    curl: &Path,
    tar: &Path,
    home: &Path,
    scratch: &Path,
    cancel: &Cancellation,
) -> Result<()> {
    install_release_with(
        curl,
        tar,
        home,
        scratch,
        cancel,
        release_asset(std::env::consts::ARCH)?,
    )
}
fn install_release_with(
    curl: &Path,
    tar: &Path,
    home: &Path,
    scratch: &Path,
    cancel: &Cancellation,
    asset: (&str, &str),
) -> Result<()> {
    let (root, digest) = asset;
    let archive = scratch.join("omaspeak.tar.xz");
    let mut command = ProcessCommand::new(curl);
    command
        .args([
            "--fail",
            "--location",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "--max-time",
            "180",
            "--max-filesize",
            "67108864",
            "--output",
        ])
        .arg(&archive)
        .arg(format!(
            "https://github.com/jacob-vincent-mink/omaspeak/releases/download/v0.0.1/{root}.tar.xz"
        ));
    eprintln!("Downloading Omaspeak 0.0.1 (model setup follows separately)…");
    run(&mut command, scratch, cancel, Duration::from_secs(190))?;
    install_archive(&archive, digest, root, tar, home, scratch, cancel)
}
#[allow(clippy::too_many_arguments)]
fn install_archive(
    archive: &Path,
    digest: &str,
    root: &str,
    tar: &Path,
    home: &Path,
    scratch: &Path,
    cancel: &Cancellation,
) -> Result<()> {
    use sha2::{Digest, Sha256};
    anyhow::ensure!(
        fs::metadata(archive)?.len() <= 64 * 1024 * 1024,
        "Omaspeak archive exceeds size limit"
    );
    anyhow::ensure!(
        format!("{:x}", Sha256::digest(fs::read(archive)?)) == digest,
        "Omaspeak release checksum mismatch; nothing installed"
    );
    let unpacked = scratch.join("release");
    fs::create_dir(&unpacked)?;
    let mut command = ProcessCommand::new(tar);
    command
        .arg("-xJf")
        .arg(archive)
        .arg("-C")
        .arg(&unpacked)
        .args(["--no-same-owner", "--no-same-permissions"]);
    run(&mut command, scratch, cancel, Duration::from_secs(60))?;
    let payload = unpacked.join(root);
    anyhow::ensure!(
        payload.join("omaspeak").is_file()
            && payload.join("lib/libaudiocpp.so.0.1.0").is_file()
            && payload.join("licenses").is_dir(),
        "incomplete Omaspeak release"
    );
    let destination = home.join(".local/opt").join(root);
    let launcher = home.join(".local/bin/omaspeak");
    anyhow::ensure!(
        fs::symlink_metadata(&destination).is_err() && fs::symlink_metadata(&launcher).is_err(),
        "Omaspeak destination already exists; inspect it before installing"
    );
    fs::create_dir_all(destination.parent().context("missing install parent")?)?;
    fs::create_dir_all(launcher.parent().context("missing launcher parent")?)?;
    // Copy into an unpublished sibling first: cache and home can be on different filesystems.
    let staging = destination.with_extension(format!("install-{}", std::process::id()));
    anyhow::ensure!(
        !staging.exists(),
        "unfinished installation exists; inspect {}",
        staging.display()
    );
    fs::create_dir(&staging)?;
    let result = (|| -> Result<()> {
        let mut copy = ProcessCommand::new("cp");
        copy.args(["-a", "--"]).arg(payload.join(".")).arg(&staging);
        run(&mut copy, scratch, cancel, Duration::from_secs(60))?;
        fs::rename(&staging, &destination)?;
        if let Err(error) = std::os::unix::fs::symlink(destination.join("omaspeak"), &launcher) {
            let _ = fs::remove_dir_all(&destination);
            return Err(error.into());
        }
        Ok(())
    })();
    if staging.exists() {
        let _ = fs::remove_dir_all(staging);
    }
    result?;
    eprintln!(
        "Installed {}. No systemd service was enabled.",
        launcher.display()
    );
    Ok(())
}

#[derive(Clone, Deserialize)]
struct Voice {
    id: i32,
    name: String,
}

fn voices(binary: &Path, scratch: &Path, cancel: &Cancellation) -> Result<Vec<Voice>> {
    let mut command = ProcessCommand::new(binary);
    command.args(["voices", "--json"]);
    let output = run(&mut command, scratch, cancel, Duration::from_secs(30))?;
    let voices: Vec<Voice> = serde_json::from_slice(&output)
        .context("read Omaspeak voice inventory; run omaspeak setup first")?;
    anyhow::ensure!(
        !voices.is_empty() && voices.len() <= 128,
        "Omaspeak returned an empty or oversized voice inventory"
    );
    Ok(voices)
}

fn choose(title: &str, help: &str, items: &[MenuItem]) -> Result<usize> {
    select(title, help, items, 0)?.context("onboarding cancelled; configuration unchanged")
}

/// All generated data stays in a private owned SampleSet. Nothing touches the
/// human calibration/validation lists, even when synthesis or review fails.
pub(super) fn augment(
    dataset: &mut Dataset,
    phrase: &str,
    paths: &AppPaths,
    cancel: &Cancellation,
    validate: &mut dyn FnMut(&[f32]) -> Result<()>,
) -> Result<SampleSet> {
    let binary = executable("omaspeak")
        .context("Omaspeak is not installed; return to onboarding to install it")?;
    augment_with(
        &binary,
        dataset,
        phrase,
        paths,
        cancel,
        validate,
        super::onboarding::text_input,
        review_pronunciation,
    )
}
#[allow(clippy::too_many_arguments)]
fn augment_with(
    binary: &Path,
    dataset: &mut Dataset,
    phrase: &str,
    paths: &AppPaths,
    cancel: &Cancellation,
    validate: &mut dyn FnMut(&[f32]) -> Result<()>,
    mut input: impl FnMut(&str) -> Result<String>,
    mut approve: impl FnMut(&Voice, &str, &Path, &Path, &Cancellation) -> Result<PronunciationReview>,
) -> Result<SampleSet> {
    let mut generated = SampleSet::create(paths)?;
    let scratch = generated.directory().to_owned();
    let inventory = loop {
        match voices(binary, &scratch, cancel) {
            Ok(voices) => break voices,
            Err(error) => {
                if choose(
                    "Set up Omaspeak",
                    &format!("{error:#}"),
                    &[
                        MenuItem::available(
                            "Open Omaspeak setup",
                            "Choose a local model and runtime, then return to enrollment",
                        ),
                        MenuItem::available("Cancel", "Your saved human dataset remains available"),
                    ],
                )? != 0
                {
                    bail!("Omaspeak setup cancelled");
                }
                anyhow::ensure!(
                    ProcessCommand::new(binary).arg("setup").status()?.success(),
                    "Omaspeak setup failed"
                );
            }
        }
    };
    let text = input(&format!(
        "TTS pronunciation text for {phrase:?} (use the phrase or a phonetic spelling)"
    ))?;
    let negatives = [
        input("First similar-sounding phrase that must NOT trigger")?,
        input("Second similar-sounding phrase that must NOT trigger")?,
        input("One ordinary sentence that must NOT trigger")?,
    ];
    validate_texts(phrase, &text, &negatives)?;
    let positions: BTreeSet<_> = [0, inventory.len() / 2, inventory.len() - 1]
        .into_iter()
        .collect();
    let mut additions = Vec::new();
    for position in positions {
        cancel.check()?;
        let voice = &inventory[position];
        let preview = scratch.join("pronunciation.wav");
        let mut voice_text = text.clone();
        let approved = loop {
            loop {
                match synthesize(binary, voice, &voice_text, 1.0, &preview, &scratch, cancel) {
                    Ok(()) => break,
                    Err(error) => {
                        cancel.check()?;
                        if choose(
                            "Omaspeak needs setup",
                            &format!("{error:#}"),
                            &[
                                MenuItem::available(
                                    "Open Omaspeak setup",
                                    "Configure its local model/runtime, then retry this preview",
                                ),
                                MenuItem::available(
                                    "Cancel assistance",
                                    "Keep the saved human dataset",
                                ),
                            ],
                        )? != 0
                        {
                            return Err(error);
                        }
                        anyhow::ensure!(
                            ProcessCommand::new(binary).arg("setup").status()?.success(),
                            "Omaspeak setup failed"
                        );
                    }
                }
            }
            match approve(voice, &voice_text, &preview, &scratch, cancel)? {
                PronunciationReview::Approve => break true,
                PronunciationReview::Skip => break false,
                PronunciationReview::Edit => {
                    loop {
                        let replacement = input(&format!(
                            "Pronunciation for voice {} (current: {:?}; other voices keep their own spelling)",
                            voice.name, voice_text
                        ))?;
                        match validate_texts(phrase, &replacement, &negatives) {
                            Ok(()) => {
                                voice_text = replacement;
                                break;
                            }
                            Err(error) => eprintln!("Spelling not changed: {error:#}"),
                        }
                    }
                    // Regeneration returns to a fresh review: prior playback never
                    // authorizes an edited pronunciation.
                }
            }
        };
        if !approved {
            continue;
        }
        for (positive, utterance, speed) in plan(&voice_text, &negatives) {
            cancel.check()?;
            eprintln!(
                "Generating {} example with {} at {:.1}×…",
                if positive {
                    "wake-phrase"
                } else {
                    "other-speech"
                },
                voice.name,
                speed
            );
            let output = scratch.join("generated.wav");
            synthesize(binary, voice, utterance, speed, &output, &scratch, cancel)?;
            generated.import(&output)?;
            let audio_path = generated
                .files
                .last()
                .context("missing generated recording")?
                .clone();
            let (_, audio) = crate::engine::audio::read_wave(&audio_path)?;
            if let Err(error) = validate(&audio) {
                // Reject a split/silent synthetic clip, not the user's whole session.
                let removed = generated
                    .files
                    .pop()
                    .context("missing rejected recording")?;
                fs::remove_file(removed)?;
                eprintln!("Skipped generated clip: {error:#}");
                continue;
            }
            additions.push(LabeledRecording {
                audio: audio_path,
                positive,
                generated: Some(GeneratedRecording {
                    generator: "omaspeak".into(),
                    voice: voice.name.clone(),
                    text: utterance.to_owned(),
                    speed,
                }),
            });
        }
    }
    anyhow::ensure!(
        additions.iter().any(|r| r.positive) && additions.iter().any(|r| !r.positive),
        "no usable approved positive/negative synthetic pair; human recordings remain available"
    );
    anyhow::ensure!(
        dataset.training.len() + additions.len() <= 128,
        "too many training examples after augmentation"
    );
    dataset.training.extend(additions);
    Ok(generated)
}

fn plan<'a>(positive: &'a str, negatives: &'a [String]) -> Vec<(bool, &'a str, f32)> {
    [0.9, 1.0, 1.1]
        .into_iter()
        .map(|speed| (true, positive, speed))
        .chain(negatives.iter().map(|text| (false, text.as_str(), 1.0)))
        .collect()
}

fn validate_texts(phrase: &str, positive: &str, negatives: &[String]) -> Result<()> {
    anyhow::ensure!(
        positive.chars().count() <= 160 && negatives.iter().all(|text| text.chars().count() <= 160),
        "keep each synthetic phrase under 160 characters"
    );
    let phrase = crate::phrase::normalize_tokens(phrase);
    let positive = crate::phrase::normalize_tokens(positive);
    anyhow::ensure!(
        !phrase.is_empty() && !positive.is_empty(),
        "wake phrase and pronunciation text cannot be empty"
    );
    let mut seen = BTreeSet::new();
    for text in negatives {
        let tokens = crate::phrase::normalize_tokens(text);
        anyhow::ensure!(
            !tokens.is_empty() && seen.insert(tokens.clone()),
            "negative phrases must be nonempty and distinct"
        );
        anyhow::ensure!(
            !tokens.windows(phrase.len()).any(|w| w == phrase)
                && !tokens.windows(positive.len()).any(|w| w == positive),
            "negative phrase contains the wake phrase or its TTS pronunciation text"
        );
    }
    Ok(())
}

fn synthesize(
    binary: &Path,
    voice: &Voice,
    text: &str,
    speed: f32,
    output: &Path,
    scratch: &Path,
    cancel: &Cancellation,
) -> Result<()> {
    // Remove the previous result so a successful but incompatible command cannot
    // accidentally reuse stale audio. Text is an argv value, never shell code.
    if output.exists() {
        fs::remove_file(output)?;
    }
    let mut command = ProcessCommand::new(binary);
    command
        .arg("say")
        .args(["--no-play", "--out"])
        .arg(output)
        .arg("--voice")
        .arg(voice.id.to_string())
        .arg("--speed")
        .arg(speed.to_string())
        .arg("--")
        .arg(text);
    run(&mut command, scratch, cancel, Duration::from_secs(180))?;
    anyhow::ensure!(
        output.is_file(),
        "Omaspeak did not write the requested WAV; run omaspeak setup"
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum PronunciationReview {
    Approve,
    Skip,
    Edit,
}

fn review_pronunciation(
    voice: &Voice,
    text: &str,
    preview: &Path,
    scratch: &Path,
    cancel: &Cancellation,
) -> Result<PronunciationReview> {
    let player = executable("pw-play").or_else(|| executable("aplay"));
    review_with(voice, text, player.is_some(), choose, || {
        let mut command = ProcessCommand::new(player.as_ref().context("no audio player")?);
        command.arg(preview);
        run(&mut command, scratch, cancel, Duration::from_secs(45))?;
        Ok(())
    })
}
fn review_with(
    voice: &Voice,
    text: &str,
    has_player: bool,
    mut ask: impl FnMut(&str, &str, &[MenuItem]) -> Result<usize>,
    mut play: impl FnMut() -> Result<()>,
) -> Result<PronunciationReview> {
    let mut heard = false;
    loop {
        let play_item = if has_player {
            MenuItem::available(
                "Play pronunciation",
                "Play this one preview through your speakers",
            )
        } else {
            MenuItem::unavailable(
                "Play pronunciation",
                "Install pw-play or aplay to review generated speech",
            )
        };
        let approve = if heard {
            MenuItem::available(
                "Approve this voice",
                "It pronounces the entire wake phrase correctly",
            )
        } else {
            MenuItem::unavailable("Approve this voice", "Listen before approving this voice")
        };
        match ask(
            &format!("Review voice {}", voice.name),
            &format!(
                "Pronunciation text: {text:?}\nOnly explicitly played previews make sound; generation itself is silent."
            ),
            &[
                play_item,
                approve,
                MenuItem::available("Skip voice", "Do not use examples from this voice"),
                MenuItem::available(
                    "Edit pronunciation",
                    "Change this voice's spelling and regenerate its preview",
                ),
            ],
        )? {
            0 => {
                anyhow::ensure!(has_player, "no audio player");
                play()?;
                heard = true;
            }
            1 if heard => return Ok(PronunciationReview::Approve),
            2 => return Ok(PronunciationReview::Skip),
            3 => return Ok(PronunciationReview::Edit),
            _ => bail!("pronunciation approval requires playback"),
        }
    }
}

fn run(
    command: &mut ProcessCommand,
    scratch: &Path,
    cancel: &Cancellation,
    timeout: Duration,
) -> Result<Vec<u8>> {
    let output = scratch.join("process.stdout");
    let errors = scratch.join("process.stderr");
    let private = |path: &Path| {
        fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .mode(0o600)
            .open(path)
    };
    let mut child = command
        .stdin(Stdio::null())
        .stdout(private(&output)?)
        .stderr(private(&errors)?)
        .process_group(0)
        .spawn()
        .context("start Omaspeak integration command")?;
    let deadline = Instant::now() + timeout;
    loop {
        if cancel.check().is_err() || Instant::now() >= deadline {
            // Owned process group includes transient inference/playback workers.
            unsafe {
                libc::kill(-(child.id() as i32), libc::SIGKILL);
            }
            let _ = child.wait();
            bail!("Omaspeak operation cancelled or timed out");
        }
        if let Some(status) = child.try_wait()? {
            anyhow::ensure!(
                fs::metadata(&output)?.len() <= 1024 * 1024
                    && fs::metadata(&errors)?.len() <= 1024 * 1024,
                "Omaspeak output exceeded limit"
            );
            anyhow::ensure!(
                status.success(),
                "Omaspeak operation failed: {}",
                fs::read_to_string(&errors)?
            );
            return Ok(fs::read(&output)?);
        }
        thread::sleep(Duration::from_millis(50));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{isolated_paths, unique_directory};
    use std::cell::Cell;

    #[test]
    fn install_is_optional_and_never_runs_during_discovery_or_decline() {
        assert!(
            prepare_with(
                |_| Some("present".into()),
                |_, _, _| panic!("no prompt"),
                |_, _| panic!("no install")
            )
            .unwrap()
        );
        assert!(
            !prepare_with(
                |_| None,
                |_, _, items| {
                    assert!(!items[1].enabled);
                    Ok(0)
                },
                |_, _| panic!("no install")
            )
            .unwrap()
        );
        let installed = Cell::new(false);
        assert!(
            prepare_with(
                |name| if name != "omaspeak" || installed.get() {
                    Some(name.into())
                } else {
                    None
                },
                |_, _, items| {
                    assert!(items[1].enabled);
                    Ok(1)
                },
                |pacman, sudo| {
                    assert_eq!(pacman, Path::new("curl"));
                    assert_eq!(sudo, Path::new("tar"));
                    installed.set(true);
                    Ok(())
                }
            )
            .unwrap()
        );
        assert!(
            prepare_with(
                |name| (name != "omaspeak").then(|| name.into()),
                |_, _, _| Ok(1),
                |_, _| anyhow::bail!("installation declined")
            )
            .is_err()
        );
        assert!(
            prepare_with(
                |_| None,
                |_, _, _| anyhow::bail!("cancel"),
                |_, _| panic!("cancel must not install")
            )
            .is_err()
        );
        assert!(
            prepare_with(
                |name| (name != "omaspeak").then(|| name.into()),
                |_, _, _| Ok(1),
                |_, _| Ok(())
            )
            .unwrap_err()
            .to_string()
            .contains("still missing")
        );
        assert!(executable("sh").is_some());
        assert!(executable("omawake-nonexistent-test-tool").is_none());
    }
    #[test]
    fn release_install_verifies_checksum_and_keeps_binary_with_its_library() {
        use sha2::{Digest, Sha256};
        let root = unique_directory("omaspeak-install", "verified");
        let payload = root.join("fixture");
        fs::create_dir_all(payload.join("lib")).unwrap();
        fs::create_dir(payload.join("licenses")).unwrap();
        fs::write(payload.join("omaspeak"), "#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(payload.join("omaspeak"), fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(payload.join("lib/libaudiocpp.so.0.1.0"), "fixture library").unwrap();
        let archive = root.join("archive.tar.xz");
        assert!(
            ProcessCommand::new("tar")
                .arg("-cJf")
                .arg(&archive)
                .arg("-C")
                .arg(&root)
                .arg("fixture")
                .status()
                .unwrap()
                .success()
        );
        let digest = format!("{:x}", Sha256::digest(fs::read(&archive).unwrap()));
        let home = root.join("home");
        let scratch = root.join("scratch");
        fs::create_dir(&scratch).unwrap();
        let cancel = Cancellation::new().unwrap();
        assert!(
            install_archive(
                &archive,
                "wrong",
                "fixture",
                Path::new("/usr/bin/tar"),
                &home,
                &scratch,
                &cancel
            )
            .unwrap_err()
            .to_string()
            .contains("checksum")
        );
        assert!(!home.exists());
        let curl = root.join("curl");
        fs::write(&curl, format!("#!/usr/bin/python3\nimport sys,shutil\nargs=sys.argv[1:]\nassert '--proto' in args and '=https' in args and '--max-filesize' in args\nassert args[-1].startswith('https://github.com/jacob-vincent-mink/omaspeak/releases/')\nshutil.copyfile({:?}, args[args.index('--output')+1])\n", archive.to_str().unwrap())).unwrap();
        fs::set_permissions(&curl, fs::Permissions::from_mode(0o755)).unwrap();
        install_release_with(
            &curl,
            Path::new("/usr/bin/tar"),
            &home,
            &scratch,
            &cancel,
            ("fixture", &digest),
        )
        .unwrap();
        assert_eq!(
            fs::read_link(home.join(".local/bin/omaspeak")).unwrap(),
            home.join(".local/opt/fixture/omaspeak")
        );
        assert!(
            home.join(".local/opt/fixture/lib/libaudiocpp.so.0.1.0")
                .is_file()
        );
        assert!(home.join(".local/opt/fixture/licenses").is_dir());
        assert!(!home.join(".config/systemd").exists());
        assert!(release_asset("x86_64").is_ok());
        assert!(release_asset("aarch64").is_ok());
        assert!(release_asset("unknown").is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn pronunciation_approval_requires_explicit_playback_and_allows_skipping() {
        let voice = Voice {
            id: 7,
            name: "voice".into(),
        };
        let stage = Cell::new(0);
        let played = Cell::new(false);
        assert!(
            review_with(
                &voice,
                "original",
                true,
                |_, _, items| {
                    let n = stage.get();
                    stage.set(n + 1);
                    assert_eq!(items[1].enabled, n > 0);
                    Ok(n)
                },
                || {
                    played.set(true);
                    Ok(())
                }
            )
            .unwrap()
                == PronunciationReview::Approve
        );
        assert!(played.get());
        assert!(
            review_with(
                &voice,
                "original",
                false,
                |_, _, items| {
                    assert!(!items[0].enabled);
                    Ok(2)
                },
                || panic!("no playback")
            )
            .unwrap()
                == PronunciationReview::Skip
        );
        assert!(
            review_with(
                &voice,
                "original",
                true,
                |_, _, _| Ok(1),
                || panic!("approval cannot bypass playback")
            )
            .is_err()
        );
        assert!(
            review_with(
                &voice,
                "original",
                true,
                |_, _, _| Ok(0),
                || anyhow::bail!("player failed")
            )
            .is_err()
        );
    }
    fn fixture() -> (PathBuf, AppPaths, SampleSet, Dataset, PathBuf) {
        let root = unique_directory("omaspeak-assistance", "dataset");
        let paths = isolated_paths(&root);
        let mut audio = SampleSet::create(&paths).unwrap();
        let mut splits = Vec::new();
        for role in 0..3 {
            let mut split = Vec::new();
            for n in 0..4 {
                audio
                    .push(&vec![0.1 + (role * 4 + n) as f32 * 0.001; 4000])
                    .unwrap();
                split.push(LabeledRecording {
                    audio: audio.files.last().unwrap().clone(),
                    positive: n < 2,
                    generated: None,
                });
            }
            splits.push(split);
        }
        let dataset = Dataset {
            training: splits.remove(0),
            calibration: splits.remove(0),
            validation: splits.remove(0),
        };
        let fake = root.join("omaspeak");
        fs::write(
            &fake,
            format!(
                r##"#!/usr/bin/python3
import sys,json,shutil
from pathlib import Path
args=sys.argv[1:]
with open({log:?},'a') as f:f.write(json.dumps(args)+'\n')
if args==['voices','--json']:
 print(json.dumps([{{'id':7,'name':'M1'}},{{'id':9,'name':'F1'}},{{'id':13,'name':'M2'}}]))
else:
 assert args[0]=='say' and '--no-play' in args and '--temperature' not in args
 assert args[-2]=='--'
 shutil.copyfile({source:?},args[args.index('--out')+1])
"##,
                log = root.join("argv.jsonl").to_str().unwrap(),
                source = audio.files[0].to_str().unwrap()
            ),
        )
        .unwrap();
        fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
        (root, paths, audio, dataset, fake)
    }
    fn input() -> impl FnMut(&str) -> Result<String> {
        let mut words = ["Hey unusual", "Hello there", "Are you", "Close the door"].into_iter();
        move |_| Ok(words.next().unwrap().into())
    }
    #[test]
    fn synthesis_is_silent_labeled_and_only_augments_training() {
        let (root, paths, source, mut dataset, fake) = fixture();
        let cal = serde_json::to_vec(&dataset.calibration).unwrap();
        let held = serde_json::to_vec(&dataset.validation).unwrap();
        let generated = augment_with(
            &fake,
            &mut dataset,
            "Hey unusual",
            &paths,
            &Cancellation::new().unwrap(),
            &mut |_| Ok(()),
            input(),
            |_, _, _, _, _| Ok(PronunciationReview::Approve),
        )
        .unwrap();
        assert_eq!(dataset.training.len(), 22);
        assert_eq!(generated.files.len(), 18);
        assert_eq!(serde_json::to_vec(&dataset.calibration).unwrap(), cal);
        assert_eq!(serde_json::to_vec(&dataset.validation).unwrap(), held);
        assert_eq!(
            dataset
                .training
                .iter()
                .filter(|r| r.generated.is_some() && r.positive)
                .count(),
            9
        );
        let checkpoint = training::retain_dataset(&paths, "unusual", &dataset).unwrap();
        drop(generated);
        drop(source);
        let retained = Dataset::load(&checkpoint.join("manifest.json")).unwrap();
        assert_eq!(
            retained
                .training
                .iter()
                .filter(|r| r.generated.is_some())
                .count(),
            18
        );
        assert!(retained.training.iter().all(|r| r.audio.is_file()));
        let mut invalid = retained.clone();
        invalid.validation[0].generated = retained.training[4].generated.clone();
        fs::write(
            root.join("invalid.json"),
            serde_json::to_vec(&invalid).unwrap(),
        )
        .unwrap();
        assert!(
            Dataset::load(&root.join("invalid.json"))
                .unwrap_err()
                .to_string()
                .contains("never calibration or validation")
        );
        let log = fs::read_to_string(root.join("argv.jsonl")).unwrap();
        assert!(log.contains("--no-play"));
        assert!(!log.contains("--temperature"));
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn edited_spellings_are_regenerated_and_applied_only_to_the_selected_voice() {
        let (root, paths, source, mut dataset, fake) = fixture();
        let mut inputs = [
            "Hey unusual",
            "Hello there",
            "Are you",
            "Close the door",
            "Are you",
            "Hay un yoo shul",
            "Hey un yoo shul",
        ]
        .into_iter();
        let mut reviewed = Vec::new();
        let mut edits = 0;
        let generated = augment_with(
            &fake,
            &mut dataset,
            "Hey unusual",
            &paths,
            &Cancellation::new().unwrap(),
            &mut |_| Ok(()),
            |_| Ok(inputs.next().unwrap().to_owned()),
            |voice, text, preview, _, _| {
                assert!(preview.is_file());
                reviewed.push((voice.id, text.to_owned()));
                if voice.id == 7 && edits < 2 {
                    edits += 1;
                    Ok(PronunciationReview::Edit)
                } else {
                    Ok(PronunciationReview::Approve)
                }
            },
        )
        .unwrap();
        assert_eq!(
            &reviewed[..3],
            &[
                (7, "Hey unusual".into()),
                (7, "Hay un yoo shul".into()),
                (7, "Hey un yoo shul".into())
            ]
        );
        assert!(reviewed[3..].iter().all(|(_, text)| text == "Hey unusual"));
        for row in dataset.training.iter().filter(|row| row.positive) {
            if let Some(origin) = &row.generated {
                assert_eq!(
                    origin.text,
                    if origin.voice == "M1" {
                        "Hey un yoo shul"
                    } else {
                        "Hey unusual"
                    }
                );
            }
        }
        let checkpoint = training::retain_dataset(&paths, "unusual", &dataset).unwrap();
        let retained = Dataset::load(&checkpoint.join("manifest.json")).unwrap();
        assert!(retained.training.iter().any(|r| {
            r.generated
                .as_ref()
                .is_some_and(|g| g.voice == "M1" && g.text == "Hey un yoo shul")
        }));
        let log = fs::read_to_string(root.join("argv.jsonl")).unwrap();
        assert!(log.contains("Hay un yoo shul"));
        assert!(log.contains("Hey un yoo shul"));
        drop(generated);
        drop(source);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn edited_previews_require_new_playback_and_edit_cancellation_keeps_human_data() {
        let voice = Voice {
            id: 7,
            name: "M1".into(),
        };
        let mut choices = [0, 3].into_iter();
        assert_eq!(
            review_with(
                &voice,
                "old spelling",
                true,
                |_, help, items| {
                    assert!(help.contains("old spelling"));
                    assert_eq!(items[3].label, "Edit pronunciation");
                    Ok(choices.next().unwrap())
                },
                || Ok(())
            )
            .unwrap(),
            PronunciationReview::Edit
        );
        assert!(
            review_with(
                &voice,
                "new spelling",
                true,
                |_, _, items| {
                    assert!(!items[1].enabled);
                    Ok(1)
                },
                || panic!("not played")
            )
            .is_err()
        );
        let (root, paths, source, mut dataset, fake) = fixture();
        let before = serde_json::to_vec(&dataset).unwrap();
        let mut first = input();
        let mut count = 0;
        let result = augment_with(
            &fake,
            &mut dataset,
            "Hey unusual",
            &paths,
            &Cancellation::new().unwrap(),
            &mut |_| Ok(()),
            |label| {
                count += 1;
                if count > 4 {
                    anyhow::bail!("onboarding cancelled");
                }
                first(label)
            },
            |_, _, _, _, _| Ok(PronunciationReview::Edit),
        );
        assert!(result.is_err());
        assert_eq!(serde_json::to_vec(&dataset).unwrap(), before);
        assert_eq!(
            fs::read_dir(paths.cache_dir.join("onboarding"))
                .unwrap()
                .count(),
            1
        );
        drop(source);
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rejected_voices_and_split_generated_clips_do_not_mutate_human_dataset() {
        for reject_voice in [false, true] {
            let (root, paths, source, mut dataset, fake) = fixture();
            let before = serde_json::to_vec(&dataset).unwrap();
            assert!(
                augment_with(
                    &fake,
                    &mut dataset,
                    "Hey unusual",
                    &paths,
                    &Cancellation::new().unwrap(),
                    &mut |_| anyhow::bail!("two segments"),
                    input(),
                    |_, _, _, _, _| Ok(if reject_voice {
                        PronunciationReview::Skip
                    } else {
                        PronunciationReview::Approve
                    })
                )
                .is_err()
            );
            assert_eq!(serde_json::to_vec(&dataset).unwrap(), before);
            assert_eq!(
                fs::read_dir(paths.cache_dir.join("onboarding"))
                    .unwrap()
                    .count(),
                1
            );
            drop(source);
            fs::remove_dir_all(root).unwrap();
        }
    }
    #[test]
    fn negative_prompt_checks_and_subprocess_errors_are_actionable() {
        assert!(
            validate_texts("Hey unusual", "Hey unusual", &["Hey unusual please".into()]).is_err()
        );
        assert!(validate_texts("Hey unusual", "Hay you", &["Hay you".into()]).is_err());
        assert!(validate_texts("Hey unusual", "Hay you", &["hi".into(), "hi".into()]).is_err());
        assert!(validate_texts("Hey unusual", "Hay you", &["".into()]).is_err());
        let root = unique_directory("omaspeak-assistance", "errors");
        let cancel = Cancellation::new().unwrap();
        let mut command = ProcessCommand::new("/usr/bin/python3");
        command.args([
            "-c",
            "import sys; print('failed',file=sys.stderr); sys.exit(3)",
        ]);
        assert!(
            run(&mut command, &root, &cancel, Duration::from_secs(5))
                .unwrap_err()
                .to_string()
                .contains("failed")
        );
        let mut command = ProcessCommand::new("/usr/bin/python3");
        command.args(["-c", "import time; time.sleep(5)"]);
        assert!(
            run(&mut command, &root, &cancel, Duration::from_millis(20))
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );
        fs::remove_dir_all(root).unwrap();
    }
}
