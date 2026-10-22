import assert from "node:assert/strict"
import { test } from "node:test"

import {
  decideLanguage,
  hintFromServerTiming,
  languageForCountry,
} from "./language.ts"

test("country VN serves Vietnamese, every other or unknown country English", () => {
  assert.equal(languageForCountry("VN"), "vi")
  assert.equal(languageForCountry("vn"), "vi")
  assert.equal(languageForCountry(" VN "), "vi")
  assert.equal(languageForCountry("US"), "en")
  assert.equal(languageForCountry("SG"), "en")
  assert.equal(languageForCountry("XX"), "en")
  assert.equal(languageForCountry(""), "en")
  assert.equal(languageForCountry(null), "en")
  assert.equal(languageForCountry(undefined), "en")
})

test("an explicit choice beats the geoip hint in both directions", () => {
  assert.equal(decideLanguage({ stored: "en", hint: "vi" }), "en")
  assert.equal(decideLanguage({ stored: "vi", hint: "en" }), "vi")
  assert.equal(
    decideLanguage({ stored: "vi", hint: "en", browser: "en-US" }),
    "vi"
  )
})

test("without a choice the hint decides, then the browser language", () => {
  assert.equal(decideLanguage({ hint: "vi", browser: "en-US" }), "vi")
  assert.equal(decideLanguage({ hint: "en", browser: "vi-VN" }), "en")
  assert.equal(decideLanguage({ browser: "vi-VN" }), "vi")
  assert.equal(decideLanguage({ browser: "fr-FR" }), "en")
  assert.equal(decideLanguage({}), "en")
})

test("a corrupt stored value or hint is ignored", () => {
  assert.equal(decideLanguage({ stored: "fr", hint: "vi" }), "vi")
  assert.equal(decideLanguage({ stored: "", hint: "zz", browser: "vi" }), "vi")
  assert.equal(decideLanguage({ stored: null, hint: null }), "en")
})

test("the hint is read from the geo Server-Timing metric of the page", () => {
  assert.equal(hintFromServerTiming([{ name: "geo", description: "VN" }]), "vi")
  assert.equal(hintFromServerTiming([{ name: "geo", description: "US" }]), "en")
  assert.equal(hintFromServerTiming([{ name: "geo", description: "T1" }]), "en")
  assert.equal(
    hintFromServerTiming([
      { name: "cfEdge", description: "" },
      { name: "geo", description: "vn" },
    ]),
    "vi"
  )
  // No rule (dev server, a missing Cloudflare rule): no hint, so the browser language decides.
  assert.equal(
    hintFromServerTiming([{ name: "cfEdge", description: "" }]),
    null
  )
  assert.equal(hintFromServerTiming([]), null)
  assert.equal(hintFromServerTiming(undefined), null)
})
