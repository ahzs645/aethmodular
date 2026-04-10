/**
 * Test script: facebook-event-scraper vs. https://www.facebook.com/trenchbrew/events
 *
 * Run with: node test-facebook-scraper.mjs
 * Requires:  npm install facebook-event-scraper
 *
 * ## Result (tested 2026-04-10)
 * Facebook returns a 301 redirect to a raw IP (57.144.174.1) that loops back on itself,
 * causing axios to hit its redirect limit. Facebook's anti-bot detection intercepts the
 * request before the library can parse any HTML.
 *
 * Status: BLOCKED by Facebook's bot detection
 */

import { scrapeFbEventList } from 'facebook-event-scraper';
import axios from 'axios';

const PAGE_URL = 'https://www.facebook.com/trenchbrew/events';

// ── 1. Library test ──────────────────────────────────────────────────────────
console.log('=== facebook-event-scraper library ===');
console.log(`Scraping: ${PAGE_URL}\n`);

try {
  const events = await scrapeFbEventList(PAGE_URL);

  if (!events || events.length === 0) {
    console.log('Result: empty — no public events found (or page is private).');
  } else {
    console.log(`Found ${events.length} event(s):\n`);
    console.log(JSON.stringify(events, null, 2));
  }
} catch (err) {
  console.error(`Library error: ${err.message}`);
}

// ── 2. Raw HTTP diagnostic ───────────────────────────────────────────────────
console.log('\n=== Raw HTTP diagnostic ===');

try {
  const resp = await axios.get(PAGE_URL, {
    headers: {
      accept: 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
      'accept-language': 'en-US,en;q=0.6',
      'user-agent':
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36',
    },
    maxRedirects: 0,
    validateStatus: () => true,
  });

  console.log(`HTTP Status : ${resp.status}`);
  console.log(`Location   : ${resp.headers['location'] ?? '(none)'}`);
  console.log(`Content-Type: ${resp.headers['content-type'] ?? '(none)'}`);
} catch (err) {
  console.error(`Raw HTTP error: ${err.message}`);
}

// ── 3. Summary ───────────────────────────────────────────────────────────────
console.log(`
=== Summary ===
URL tested : ${PAGE_URL}
Library    : facebook-event-scraper (npm)
Outcome    : BLOCKED

Root cause :
  Facebook responds with HTTP 301 → https://57.144.174.1/trenchbrew/events
  That IP then redirects back to itself, creating a redirect loop.
  axios exhausts its redirect limit; the library catches the error and
  throws a generic "Error fetching event" message, hiding the real cause.

What this means:
  Facebook's bot-detection intercepts server-side HTTP requests even with
  a realistic Chrome User-Agent. The library cannot scrape this page in
  its current form without additional circumvention (e.g., a residential
  proxy, a real browser via Playwright/Puppeteer, or a cookie with a
  logged-in session).

Alternatives to consider:
  1. Facebook Graph API  — requires app review; limited event data for pages
  2. Playwright/Puppeteer — renders JS, can handle login; higher complexity
  3. A proxy service     — ScraperAPI, Apify, etc.; adds cost
  4. Manual import       — export events from Facebook manually as iCal/CSV
`);
