import { expect, test } from "@playwright/test";

test("all eight illustration assets render across their intended pages", async ({
  page,
}) => {
  const assets = new Set<string>();
  for (const path of ["/", "/about", "/chat"]) {
    await page.goto(path);
    const images = page.locator('img[src^="/assets/visuals/"]');
    await expect(images.first()).toBeAttached();
    for (const image of await images.all()) {
      await image.scrollIntoViewIfNeeded();
      await expect
        .poll(() =>
          image.evaluate(
            (node) =>
              (node as HTMLImageElement).complete &&
              (node as HTMLImageElement).naturalWidth > 0,
          ),
        )
        .toBe(true);
      assets.add((await image.getAttribute("src")) || "");
    }
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
  expect(assets.size).toBe(8);
});

test("navigation source menu has valid destinations and closes cleanly", async ({
  page,
  isMobile,
}) => {
  await page.goto("/");
  if (isMobile) {
    await page
      .getByRole("button", { name: "Buka navigasi", exact: true })
      .click();
    const menu = page.getByRole("navigation", {
      name: "Navigasi seluler",
      exact: true,
    });
    await expect(
      menu.getByRole("link", { name: "Fatwa DSN-MUI" }),
    ).toHaveAttribute("href", "https://dsnmui.or.id/kategori/fatwa/");
    await page.keyboard.press("Escape");
    await expect(menu).toHaveCount(0);
  } else {
    const button = page.getByRole("button", {
      name: "Sumber dokumen",
      exact: true,
    });
    await button.click();
    await expect(button).toHaveAttribute("aria-expanded", "true");
    await expect(
      page.locator("#nav-source-dropdown").getByRole("link"),
    ).toHaveCount(3);
    await page.keyboard.press("Escape");
    await expect(button).toHaveAttribute("aria-expanded", "false");
    await expect(button).toBeFocused();
    await button.click();
    await page.mouse.click(25, 200);
    await expect(button).toHaveAttribute("aria-expanded", "false");
  }
});

test("landing keeps its content, links, and layout usable", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.getByRole("heading", { level: 1 })).toHaveText(
    "Sebelum menentukan akad,pahami rujukannya.",
  );
  await expect(page.getByRole("tab", { selected: true })).toHaveText(
    "Pre-order",
  );
  await expect(page.locator("footer")).toContainText("Jelajahi AMANA");
  const links = await page
    .locator("a")
    .evaluateAll((items) => items.map((item) => item.getAttribute("href")));
  expect(
    links.every(
      (href) => href && href !== "#" && !href.startsWith("javascript:"),
    ),
  ).toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  expect(errors).toEqual([]);
});

test("primary CTA opens chat, and chat returns home", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("link", { name: "Buka AMANA", exact: true }).click();
  await expect(page).toHaveURL(/\/chat$/);
  await expect(
    page.getByRole("textbox", { name: "Pertanyaan untuk AMANA", exact: true }),
  ).toBeVisible();
  await page.getByRole("link", { name: "Beranda", exact: true }).click();
  await expect(page).toHaveURL(/\/$/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "Sebelum menentukan akad",
  );
});

test("preview tabs support pointer and keyboard without making inference requests", async ({
  page,
}) => {
  const requests: string[] = [];
  page.on("request", (request) => {
    if (request.url().includes("/rag/ask")) requests.push(request.url());
  });
  await page.goto("/");
  await page.getByRole("tab", { name: "Bagi hasil", exact: true }).click();
  await expect(page.getByRole("tabpanel")).toContainText("sistem bagi hasil");
  await page
    .getByRole("tab", { name: "Bagi hasil", exact: true })
    .press("ArrowRight");
  await expect(
    page.getByRole("tab", { name: "Pembayaran bertahap", exact: true }),
  ).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("tabpanel")).toContainText(
    "membayar secara bertahap",
  );
  expect(requests).toEqual([]);
});

test("source panel can be opened and closed, with a valid source destination", async ({
  page,
}) => {
  await page.goto("/");
  await page.locator(".preview-citation").click();
  const source = page.getByRole("complementary", {
    name: "Sumber referensi contoh",
  });
  await expect(source).toBeVisible();
  await expect(
    source.getByRole("link", { name: "Buka dokumen sumber" }),
  ).toHaveAttribute("href", "https://dsnmui.or.id/kategori/fatwa/page/17/");
  await source.getByRole("button", { name: "Tutup sumber referensi" }).click();
  await expect(source).toHaveCount(0);
  await page
    .getByRole("button", { name: "Tampilkan sumber referensi" })
    .click();
  await expect(source).toBeVisible();
  await source
    .getByRole("button", { name: "Tutup sumber referensi" })
    .press("Escape");
  await expect(source).toHaveCount(0);
});

test("preview composer passes the draft to chat without auto-sending", async ({
  page,
}) => {
  let calls = 0;
  await page.route("**/rag/ask", async (route) => {
    calls += 1;
    await route.fulfill({
      json: {
        answer: "Jawaban pengujian",
        sources: [],
        support_info: {},
        generation_meta: {},
        timing: {},
      },
    });
  });
  await page.goto("/");
  await page.locator("#preview-question").fill("Apa rujukan untuk usaha saya?");
  await page
    .getByRole("button", { name: "Buka percakapan dengan pertanyaan ini" })
    .click();
  await expect(page).toHaveURL(/\/chat\?question=/);
  const input = page.getByRole("textbox", {
    name: "Pertanyaan untuk AMANA",
    exact: true,
  });
  await expect(input).toHaveValue("Apa rujukan untuk usaha saya?");
  expect(calls).toBe(0);
  await page.getByRole("button", { name: "Kirim pesan", exact: true }).click();
  await expect(
    page.getByText("Jawaban pengujian", { exact: true }),
  ).toBeVisible();
  expect(calls).toBe(1);
});

test("example accordion expands, collapses and starts a prefilled chat", async ({
  page,
}) => {
  await page.goto("/#contoh-pertanyaan");
  const question = page.getByRole("button", {
    name: /Apa yang perlu dipahami tentang pre-order/,
  });
  await question.click();
  await expect(question).toHaveAttribute("aria-expanded", "true");
  await expect(
    page.getByRole("region", { name: /Apa yang perlu dipahami/ }),
  ).toContainText("Ceritakan barang");
  await question.click();
  await expect(question).toHaveAttribute("aria-expanded", "false");
  await question.click();
  await page.getByRole("link", { name: "Gunakan pertanyaan ini" }).click();
  await expect(
    page.getByRole("textbox", { name: "Pertanyaan untuk AMANA", exact: true }),
  ).toHaveValue(/produk pre-order/);
});

test("header navigation reaches the actual section on desktop and mobile", async ({
  page,
  isMobile,
}) => {
  await page.goto("/");
  if (isMobile)
    await page
      .getByRole("button", { name: "Buka navigasi", exact: true })
      .click();
  await page
    .getByRole("navigation", {
      name: isMobile ? "Navigasi seluler" : "Navigasi utama",
      exact: true,
    })
    .getByRole("link", { name: "Cara kerja", exact: true })
    .click();
  await expect(page).toHaveURL(/#cara-kerja$/);
  await expect
    .poll(async () =>
      page
        .locator("#cara-kerja")
        .evaluate((element) => Math.round(element.getBoundingClientRect().top)),
    )
    .toBeGreaterThanOrEqual(0);
  await expect(
    page.getByRole("heading", {
      name: /Pahami jawabannya\.\s*Periksa rujukannya\./,
    }),
  ).toBeInViewport();
});

for (const info of [
  {
    name: "Tentang AMANA",
    path: "/about",
    heading: "Rujukan yang bisa dibaca kembali.",
  },
  { name: "Privasi", path: "/privacy", heading: "Privasi percakapan Anda." },
  {
    name: "Ketentuan penggunaan",
    path: "/terms",
    heading: "Gunakan sebagai awal penelusuran.",
  },
]) {
  test(`footer ${info.name} has a real page and supports direct reload`, async ({
    page,
  }) => {
    await page.goto("/");
    await page
      .locator("footer")
      .getByRole("link", { name: info.name, exact: true })
      .click();
    await expect(page).toHaveURL(new RegExp(`${info.path}$`));
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      info.heading,
    );
    await page.reload();
    await expect(page.getByRole("heading", { level: 1 })).toHaveText(
      info.heading,
    );
    await page.getByRole("link", { name: "Kembali ke beranda" }).click();
    await expect(page.getByRole("heading", { level: 1 })).toContainText(
      "Sebelum menentukan akad",
    );
  });
}

test("chat settings and new chat work on both layouts", async ({
  page,
  isMobile,
}) => {
  await page.goto("/chat?question=Pertanyaan%20awal");
  await expect(
    page.getByRole("textbox", { name: "Pertanyaan untuk AMANA", exact: true }),
  ).toHaveValue("Pertanyaan awal");
  if (isMobile)
    await page.getByRole("button", { name: "Buka menu percakapan" }).click();
  await page
    .getByRole("button", { name: "Pengaturan", exact: true })
    .filter({ visible: true })
    .click();
  await expect(
    page.getByRole("textbox", { name: "Fokus jawaban", exact: true }),
  ).toBeVisible();
  if (isMobile)
    await page.getByRole("button", { name: "Buka menu percakapan" }).click();
  await page
    .getByRole("button", { name: "Chat Baru", exact: true })
    .filter({ visible: true })
    .click();
  await expect(page).toHaveURL(/\/chat$/);
  await expect(
    page.getByRole("textbox", { name: "Pertanyaan untuk AMANA", exact: true }),
  ).toHaveValue("");
});

test("missing routes show recovery links and respect reduced motion", async ({
  page,
}) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/halaman-tidak-ada");
  await expect(
    page.getByRole("heading", { name: "Halaman ini belum ditemukan." }),
  ).toBeVisible();
  await page
    .getByRole("link", { name: "Kembali ke beranda", exact: true })
    .click();
  await expect(page.getByRole("heading", { level: 1 })).toBeVisible();
  expect(
    await page
      .locator("html")
      .evaluate((element) => getComputedStyle(element).scrollBehavior),
  ).toBe("auto");
});

test("unavailable backend gives a readable error and releases the composer", async ({
  page,
}) => {
  await page.route("**/rag/ask", (route) =>
    route.fulfill({ status: 503, json: { detail: "Pipeline disabled" } }),
  );
  await page.goto("/chat?question=Pertanyaan%20pengujian");
  await page.getByRole("button", { name: "Kirim pesan", exact: true }).click();
  await expect(
    page.getByText("Layanan jawaban belum tersedia. Silakan coba lagi nanti.", {
      exact: true,
    }),
  ).toBeVisible();
  await expect(
    page.getByRole("textbox", { name: "Pertanyaan untuk AMANA", exact: true }),
  ).toBeEnabled();
});

test("new chat cancels a pending request instead of leaking its response", async ({
  page,
  isMobile,
}) => {
  let release!: () => void;
  let received!: () => void;
  const holdResponse = new Promise<void>((resolve) => {
    release = resolve;
  });
  const requestReceived = new Promise<void>((resolve) => {
    received = resolve;
  });
  await page.route("**/rag/ask", async (route) => {
    received();
    await holdResponse;
    await route
      .fulfill({
        json: {
          answer: "Jawaban sesi lama",
          sources: [],
          support_info: {},
          generation_meta: {},
          timing: {},
        },
      })
      .catch(() => {});
  });
  await page.goto("/chat?question=Pertanyaan%20lama");
  await page.getByRole("button", { name: "Kirim pesan", exact: true }).click();
  await requestReceived;
  const cancelled = page.waitForEvent("requestfailed", (request) =>
    request.url().includes("/rag/ask"),
  );
  if (isMobile)
    await page.getByRole("button", { name: "Buka menu percakapan" }).click();
  await page
    .getByRole("button", { name: "Chat Baru", exact: true })
    .filter({ visible: true })
    .click();
  await cancelled;
  release();
  await expect(
    page.getByRole("textbox", { name: "Pertanyaan untuk AMANA", exact: true }),
  ).toHaveValue("");
  await expect(
    page.getByText("Jawaban sesi lama", { exact: true }),
  ).toHaveCount(0);
});
