"""Play every activity type in a headless browser and check it behaves.

    pip install playwright
    python3 tools/playtest.py [--levels 1,4,11,14,17] [--chrome /path/to/chrome]

For each level it runs each step, answers correctly (reading the current item from
window.__current), and checks: the step passes, choices include the right answer once,
picture choices are all different, and the page logs no errors.
"""
import argparse, asyncio, json
from pathlib import Path

from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent


async def answer(pg, cur):
    t, item = cur["type"], cur["item"]
    if t == "listen":
        await pg.wait_for_selector(".choices .pic")
        ok = await pg.evaluate("""c => { const want = DATA.pictures[c.item.p[0]] + DATA.pictures[c.item.p[1]];
            const b = [...document.querySelectorAll('.choices .pic')];
            const hits = b.filter(x => x.textContent === want); if (hits.length) hits[0].click(); return hits.length; }""", cur)
        return ["listen: right answer shown %d times" % ok] if ok != 1 else []
    if t == "sounds":
        await pg.wait_for_selector(".egg")
        labels = await pg.eval_on_selector_all(".egg", "els => els.map(e => e.getAttribute('aria-label'))")
        want = "Letter " + item["g"]
        await pg.click(f'.egg[aria-label="{want}"]')
        return [] if labels.count(want) == 1 else [f"eggs: {labels} for {want}"]
    if t == "spell":
        await pg.wait_for_selector(".flower")
        for tile in item["t"]:
            await pg.locator(f'.flower[aria-label="Letter {tile["g"]}"]:not(.used)').first.click()
            await pg.wait_for_timeout(120)
        return []
    if t == "read":
        await pg.wait_for_selector(".frog.reader")
        for _ in range(12):
            if await pg.locator(".yn:not([hidden])").count():
                break
            await pg.click(".frog.reader")
            await pg.wait_for_timeout(120)
        match = await pg.evaluate("c => document.querySelector('.show').textContent === c.item.yes", cur)
        await pg.click('[aria-label="%s"]' % ("Yes, it matches" if match else "No, it does not match"))
        return []
    if t == "blend":
        await pg.wait_for_selector(".frog")
        for _ in range(len(item["t"])):
            await pg.click(".pond .frog")
            await pg.wait_for_timeout(150)
        await pg.wait_for_selector(".choices .pic, .choices .big-btn", timeout=6000)
        if await pg.locator(".choices .big-btn").count():
            await pg.click(".choices .big-btn")
            await pg.wait_for_selector('[aria-label="Yes, I said it"]')
            await pg.click('[aria-label="Yes, I said it"]')
            return []
        pics = await pg.eval_on_selector_all(".choices .pic", "els => els.map(e => e.textContent)")
        want = await pg.evaluate("c => DATA.pictures[c.item.w]", cur)
        issues = []
        if pics.count(want) != 1:
            issues.append(f"blend {item['w']}: right picture shown {pics.count(want)} times")
        if len(set(pics)) != len(pics):
            issues.append(f"blend {item['w']}: repeated pictures {pics}")
        await pg.locator(".choices .pic", has_text=want).first.click()
        return issues
    return [f"unknown activity {t}"]


async def run(levels, chrome):
    issues = []
    async with async_playwright() as p:
        kw = {"executable_path": chrome} if chrome else {}
        b = await p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required"], **kw)
        ctx = await b.new_context(viewport={"width": 1024, "height": 768}, has_touch=True, reduced_motion="reduce")
        await ctx.route("https://fonts.googleapis.com/**", lambda r: r.abort())
        pg = await ctx.new_page()
        pg.set_default_timeout(10000)
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        await pg.goto((ROOT / "index.html").as_uri(), wait_until="domcontentloaded")
        # sounds play at 4x speed so the run is quick
        await pg.evaluate("""() => { const orig = AudioBufferSourceNode.prototype.start;
            AudioBufferSourceNode.prototype.start = function (...a) { this.playbackRate.value = 4; return orig.apply(this, a); }; }""")
        for n in levels:
            stages = await pg.evaluate(f"stagesFor(LEVELS[{n - 1}])")
            for i, stage in enumerate(stages):
                await pg.evaluate(f"void runStage({n}, '{stage}', {i})")
                seen = 0
                while True:
                    await pg.wait_for_timeout(250)
                    if await pg.locator(".result, .stage .big-btn:has-text('Keep climbing')").count():
                        break
                    cur = await pg.evaluate("window.__current")
                    key = json.dumps(cur, sort_keys=True)
                    try:
                        issues += [f"L{n} {stage}: {x}" for x in await answer(pg, cur)]
                    except Exception as e:  # a stuck activity is a finding, not a crash
                        issues.append(f"L{n} {stage}: stuck on {cur['type']} {cur['item'].get('w', cur['item'].get('g', ''))}: {str(e)[:80]}")
                        break
                    seen += 1
                    for _ in range(60):
                        await pg.wait_for_timeout(100)
                        nxt = await pg.evaluate("window.__current")
                        if json.dumps(nxt, sort_keys=True) != key or await pg.locator("text=Keep climbing").count():
                            break
                    if seen > 30:
                        issues.append(f"L{n} {stage}: never finished")
                        break
                passed = await pg.locator("text=Keep climbing").count()
                if not passed:
                    issues.append(f"L{n} {stage}: did not pass with all answers right")
                print(f"L{n} {stage}: {'passed' if passed else 'FAILED'} after {seen} items", flush=True)
        issues += [f"page error: {e}" for e in errors]
        await b.close()
    print(f"\n{len(issues)} issues")
    for i in issues:
        print("  " + i)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,11,14,17")
    ap.add_argument("--chrome", default="/opt/pw-browsers/chromium-1194/chrome-linux/chrome")
    a = ap.parse_args()
    asyncio.run(run([int(x) for x in a.levels.split(",")], a.chrome))
