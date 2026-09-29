# Agent trên recommend_test.json: ràng buộc cứng và độ hợp ngữ cảnh

- `precision` = trung bình theo case của tỉ lệ phim đạt; `all pass` = tỉ lệ case mà mọi phim đều đạt; `hit rate` = tỉ lệ case có ≥ 1 phim đạt. Semantic chỉ tính các case có `semantic_requirement: true`.
- Judge semantic: LLM, cache trong `.eval_cache/` (dùng chung giữa các luồng).

| luồng | case có gợi ý | phim / case | case 0 phim | lỗi agent | ràng buộc: phim đạt | ràng buộc: all pass | semantic: phim đạt | semantic: precision | semantic: hit rate | semantic: NDCG |
|---|---|---|---|---|---|---|---|---|---|---|
| pass_first | 59 / 60 | 5.2 | 1 | 1 | 95.9% | 96.6% | 78.4% (171 phim) | 79.6% | 100.0% | 89.3% |

## Số case có ≥ n phim đạt

“cả hai” = đạt ràng buộc cứng và đạt semantic (hoặc case không yêu cầu semantic). Semantic tính trên các case có yêu cầu semantic.

| luồng | tiêu chí | ≥ 1 phim | ≥ 2 phim | ≥ 3 phim | ≥ 4 phim | ≥ 5 phim |
|---|---|---|---|---|---|---|
| pass_first | ràng buộc | 58 (96.7%) | 58 (96.7%) | 57 (95.0%) | 57 (95.0%) | 57 (95.0%) |
| pass_first | semantic | 32 (97.0%) | 31 (93.9%) | 29 (87.9%) | 25 (75.8%) | 15 (45.5%) |
| pass_first | cả hai | 58 (96.7%) | 57 (95.0%) | 54 (90.0%) | 51 (85.0%) | 41 (68.3%) |

## pass_first

Case không có phim: similar_to_groundhog_day_no_romance.

Phim trượt ràng buộc cứng:

- `horror_after_2010` #1 Shining, The: movie year does not meet min_year=2010
- `horror_after_2010` #2 Twilight Zone: The Movie: movie year does not meet min_year=2010
- `horror_after_2010` #3 Wes Craven's New Nightmare (Nightmare on Elm Street Part 7: Freddy's Finale, A): movie year does not meet min_year=2010
- `horror_after_2010` #4 Blue Velvet: missing required genres: ['Horror']; movie year does not meet min_year=2010
- `horror_after_2010` #5 Halloween: movie year does not meet min_year=2010
- `horror_after_2010` #6 Friday the 13th: movie year does not meet min_year=2010
- `horror_after_2010` #7 Tales from the Darkside: The Movie: movie year does not meet min_year=2010
- `horror_after_2010` #8 Scream: movie year does not meet min_year=2010
- `horror_after_2010` #9 Fright Night: movie year does not meet min_year=2010
- `horror_after_2010` #10 Poltergeist: movie year does not meet min_year=2010
- `unreliable_narrator_mystery` #2 Strangers on a Train: missing required genres: ['Mystery']
- `unreliable_narrator_mystery` #3 Suture: missing required genres: ['Mystery']
- `unreliable_narrator_mystery` #4 One Flew Over the Cuckoo's Nest: missing required genres: ['Mystery']

Phim trượt semantic:

- `toy_story_live_action` #2 American History X: American History X is live-action, but it is a dark crime drama about neo-Nazism and violence, not a family-friendly adventure/comedy comparable to Toy Story.
- `dark_psychological_thriller_twist` #1 Day the Earth Stood Still, The: The movie is a classic sci-fi drama/thriller about an alien visitor and a global warning, not a dark psychological thriller with a clever twist as requested.
- `dark_psychological_thriller_twist` #2 Star Trek II: The Wrath of Khan: Star Trek II is an action/adventure sci-fi thriller, not a dark psychological thriller; its plot focuses on space combat, revenge, and sacrifice rather than psychological tension or a twist-driven dark thriller narrative.
- `dark_psychological_thriller_twist` #3 Star Trek: First Contact: The user requested a dark psychological thriller with a clever twist, but Star Trek: First Contact is an action/adventure sci-fi film about the Borg, not a dark psychological thriller.
- `dark_psychological_thriller_twist` #4 North by Northwest: North by Northwest is a high-quality Hitchcock thriller with mistaken identity and a twist, but it is primarily an action-adventure espionage/romance film, not a dark psychological thriller as requested.
- `explain_then_recommend_similar` #4 Clue: Clue is a comedic, slapstick whodunit with multiple endings, but it lacks the strong dramatic core and morally complex character treatment implied by the assistant's stated reasons for recommending a smart mystery.
- `uplifting_drama` #4 Sweet Charity: The supplied plot focuses on an alternate happy ending, but the film kept the original stage ending, which is not described as uplifting or hopeful; thus it does not clearly match the requested mood.
- `comforting_light_evening` #1 What's Eating Gilbert Grape: The movie is a heavy drama involving suicide, depression, family burden, death, and arson, so it does not fit a comforting, light evening.
- `comforting_light_evening` #4 Lost in Translation: Lost in Translation is a melancholic drama about loneliness, midlife crisis, marital strain, and a bittersweet goodbye, so it does not fit a comforting, light evening.
- `friendship_theme_no_animation` #2 Friends with Benefits: The movie is a romantic comedy about friends who begin a sexual relationship and eventually become romantically involved, not a story centered on enduring friendship.
- `friendship_theme_no_animation` #4 Say Anything...: Say Anything... is a romantic comedy-drama centered on a teenage romance between Lloyd and Diane, not on enduring friendship, so it does not fit the user's requested theme.
- `unreliable_narrator_no_horror` #1 Blue Velvet: Blue Velvet is a Drama/Mystery/Thriller and not Horror, but the plot does not indicate an unreliable narrator; it follows Jeffrey's discoveries in a largely straightforward manner, so it does not satisfy the core request.
- `unreliable_narrator_no_horror` #3 One Flew Over the Cuckoo's Nest: The movie is a Drama and not Horror, but the supplied plot does not indicate an unreliable narrator; it presents a straightforward narrative about institutional conflict.
- `unreliable_narrator_no_horror` #5 Breaking the Waves: The plot describes Bess's psychological problems and subjective religious beliefs, but it does not indicate an unreliable narrator; the request specifically requires an unreliable narrator.
- `date_night_romance_comedy` #3 Osmosis Jones: Although Osmosis Jones lists Comedy and Romance among its genres, it is primarily an animated action/crime/thriller about the inside of a human body, with romance only a minor subplot, so it does not fit a date-night Romance and Comedy request.
- `dark_comedy_crime` #2 Police Academy 6: City Under Siege: The movie is a Crime/Comedy, but Police Academy 6 is known for broad slapstick police humor rather than the requested darkly comic tone.
- `feel_good_sports_story` #1 Field of Dreams: Field of Dreams is a baseball-themed fantasy drama about Ray Kinsella building a field and reconciling with his father, not a feel-good movie centered on an underdog sports team overcoming challenges.
- `feel_good_sports_story` #2 Hoop Dreams: Hoop Dreams is a documentary about two individuals pursuing basketball, not a feel-good movie about an underdog sports team; it emphasizes hardship and systemic issues more than uplifting team triumph.
- `feel_good_sports_story` #6 Last Action Hero: The supplied plot involves forming a community baseball team for fathers and sons, but it does not describe an underdog sports team; its listed genres are Action, Adventure, Comedy, and Fantasy, not sports/feel-good underdog themes.
- `quiet_character_study` #1 American Beauty: American Beauty is a Drama, but it is an ensemble suburban satire with multiple intersecting plotlines rather than a quiet, intimate character study.
- `quiet_character_study` #5 Blue Velvet: Blue Velvet is a Drama/Mystery/Thriller, but its plot centers on a lurid crime mystery with violence, voyeurism, and sadomasochism, so it does not fit the requested quiet, intimate character study.
- `atmospheric_scifi` #5 Total Recall: Although Total Recall is Sci-Fi and has mysterious memory/identity elements, its genres and plot emphasize Action/Adventure and Thriller with constant chases, fights, and killings, so it does not fit the requested mysterious, atmospheric tone rather than nonstop action.
- `similar_to_before_sunrise` #1 Leaving Las Vegas: Leaving Las Vegas is a Romance/Drama, but its plot centers on alcoholism, self-destruction, prostitution, and a doomed relationship, not the intimate, conversation-driven, hopeful romance of Before Sunrise.
- `similar_to_before_sunrise` #3 Jungle Book, The: Although tagged Romance, 'Jungle Book' is primarily a family adventure/fantasy about Mowgli living among animals, with a secondary childhood-to-adult romance rather than the intimate, conversation-driven dynamic requested like Before Sunrise.
- `similar_to_before_sunrise` #4 Gone with the Wind: Gone with the Wind is an epic historical war romance, not an intimate, conversation-driven film like Before Sunrise; its plot spans years, war, and melodrama rather than focusing on a single evolving dialogue between two people.
- `similar_to_before_sunrise` #5 Ghost: Ghost is a supernatural romantic thriller centered on murder, money laundering, and ghosts, not an intimate, conversation-driven romance like Before Sunrise.
- `similar_to_before_sunrise` #6 Vertigo: Vertigo is primarily a psychological mystery/thriller with obsessive romance elements, not an intimate, conversation-driven romance like Before Sunrise.
- `another_witty_mystery` #1 Dial M for Murder: Dial M for Murder is a Crime/Mystery/Thriller about a planned murder and suspenseful investigation, with a dark tone rather than the requested playful humor and witty dialogue.
- `low_key_date_night` #2 What's Eating Gilbert Grape: The film is a heavy drama dealing with suicide, depression, family hardship, and death, which conflicts with the user's request for 'nothing too heavy.'
- `low_key_date_night` #3 Crimes and Misdemeanors: The user asked for something not too heavy for a low-key date night, but Crimes and Misdemeanors involves murder, guilt, suicide, and moral anguish, making it tonally too dark and heavy for the request.
- `low_key_date_night` #4 Peter's Friends: The user asked for something not too heavy, but Peter's Friends includes serious themes of death, loss, and personal tragedy despite being primarily a comedy.
- `low_key_date_night` #5 Crimes of the Heart: The plot centers on dysfunctional family trauma, including an abusive husband being shot, a mother's suicide, a nervous breakdown, and past resentments, which conflicts with the request for something not too heavy and low-key.
- `matrix_vibe_not_action` #2 E.T. the Extra-Terrestrial: E.T. is a family-friendly sci-fi drama about a boy befriending an alien, not a mind-bending or reality-questioning film like The Matrix.
- `matrix_vibe_not_action` #3 Grand Day Out with Wallace and Gromit, A: The candidate is an animated family comedy/adventure, not a mind-bending film like The Matrix; its genres and premise do not match the requested philosophical or reality-bending feel.
- `grief_redemption_drama` #2 Noah: Although Noah is listed as a Drama, its plot focuses on divine visions, warning humanity, and rallying the Watchers, with no evidence of dealing with grief and finding redemption afterward as requested.
- `unreliable_narrator_mystery` #2 Strangers on a Train: Strangers on a Train is a crime/thriller about a murder-swap scheme, but its plot is not structured around an untrustworthy narrator or a final reality flip; it does not match the requested unreliable-narrator mystery.
- `unreliable_narrator_mystery` #4 One Flew Over the Cuckoo's Nest: The user asked for a mystery with an untrustworthy narrator and a major end twist. One Flew Over the Cuckoo's Nest is a drama set in a mental institution and does not match the mystery/unreliable-narrator/twist requirement.
