<template>
  <div class="app-shell">
    <div class="app-layout">
      <aside class="side-menu">
        <div class="side-menu__logo">
          <h1>Realtime Desk</h1>
          <p>WS {{ wsConnected ? 'ONLINE' : 'OFFLINE' }}</p>
        </div>
        <p class="side-menu__title">메뉴 목록</p>
        <nav class="side-menu__nav">
          <button
            v-for="item in sideMenuItems"
            :key="item.key"
            class="side-menu__btn"
            :class="{ active: activeView === item.key }"
            @click="setView(item.key)"
          >
            {{ item.label }}
          </button>
        </nav>
      </aside>

      <div class="view-pane">
        <ChartView
          v-if="activeView === 'chart'"
          :symbols="displayedSymbols"
          :chartSymbol="chartSymbol"
          :apiBase="apiBase"
          :chartSignals="chartSignals"
          :chartNowcast="chartNowcast"
          :wsConnected="wsConnected"
          @update:chartSymbol="(value) => (chartSymbol = value)"
        />

        <template v-else-if="activeView === 'monitoring'">
          <header class="masthead">
            <div class="masthead__titles">
              <h1>Real-Time Bottom Detector</h1>
              <p class="muted">실시간 바텀 탐지 대시보드</p>
            </div>
            <div class="masthead__status">
              <div class="api-pill">
                <span>API</span>
                <input class="api-input" v-model="apiBase" spellcheck="false" />
              </div>
              <span class="status-chip" :class="healthDot">{{ healthText }}</span>
              <span class="muted">업데이트 {{ lastUpdatedRelative }}</span>
              <button class="btn ghost" @click="manualRefresh" :disabled="loading">동기화</button>
            </div>
          </header>

          <section class="system-strip">
            <article class="metric-card">
              <header>시스템 상태</header>
              <div class="metric-card__body">
                <p class="health-line">
                  <span class="dot" :class="healthDot"></span>
                  <span>{{ healthText }}</span>
                </p>
                <p class="muted">최근 {{ lastUpdatedRelative }}</p>
                <p class="muted" v-if="error">{{ error }}</p>
              </div>
            </article>

            <article class="metric-card">
              <header>신호 현황</header>
              <div class="metric-card__body distribution">
                <div class="dist-row">
                  <span>ACTIVE</span>
                  <div class="bar"><span :style="{ width: activeShare + '%' }"></span></div>
                  <strong>{{ activeSignals.length }}</strong>
                </div>
                <div class="dist-row">
                  <span>HIGH</span>
                  <div class="bar warn"><span :style="{ width: highConfidenceShare + '%' }"></span></div>
                  <strong>{{ highConfidenceSignals.length }}</strong>
                </div>
                <p class="muted">총 {{ displayedSymbols.length }}개 심볼</p>
              </div>
            </article>

            <article class="metric-card" v-if="stackingMeta">
              <header>스태킹 설정</header>
              <div class="metric-card__body meta-grid">
                <p><span class="label">방법</span><span>{{ stackingMeta.method || 'auto' }}</span></p>
                <p v-if="stackingMeta.method_override"><span class="label">Override</span><span>{{ stackingMeta.method_override }}</span></p>
                <p><span class="label">Threshold</span><span>{{ formatThreshold(stackingMeta.threshold) }}</span></p>
                <p><span class="label">Source</span><span>{{ stackingMeta.threshold_source || 'auto' }}</span></p>
                <p><span class="label">Models</span><span>{{ stackingMeta.used_models?.length || stackingMeta.models?.length || 0 }}</span></p>
                <p v-if="(stackingMeta.used_models && stackingMeta.used_models.length) || (stackingMeta.models && stackingMeta.models.length)">
                  <span class="label">사용</span>
                  <span class="models-line">{{ (stackingMeta.used_models || stackingMeta.models || []).map(m => String(m).toUpperCase()).join(', ') }}</span>
                </p>
              </div>
            </article>

            <article class="metric-card" v-if="entryMetaCard">
              <header>엔트리 메타</header>
              <div class="metric-card__body meta-grid">
                <p><span class="label">Win-rate</span><span>{{ entryMetaCard.win_rate == null ? '–' : formatPct(entryMetaCard.win_rate) }}</span></p>
                <p><span class="label">샘플</span><span>{{ entryMetaCard.samples }}/{{ entryMetaCard.window }}</span></p>
                <p><span class="label">Target</span><span>{{ entryMetaCard.target == null ? '–' : formatPct(entryMetaCard.target) }}</span></p>
                <p><span class="label">Th(dyn)</span><span>{{ formatThreshold(entryMetaCard.th_dynamic) }}</span></p>
                <p><span class="label">Th(side)</span><span>{{ formatThreshold(entryMetaCard.th_sidecar) }}</span></p>
                <p><span class="label">Th(env)</span><span>{{ formatThreshold(entryMetaCard.th_env) }}</span></p>
              </div>
              <div class="metric-card__body" v-if="entryWrSparkPoints">
                <p class="muted">WR 추세</p>
                <svg viewBox="0 0 120 36" preserveAspectRatio="none" class="wr-spark">
                  <polyline :points="entryWrSparkPoints" />
                </svg>
              </div>
            </article>

            <article class="metric-card" v-if="trainerMeta">
              <header>트레이너</header>
              <div class="metric-card__body meta-grid">
                <p>
                  <span class="label">Heartbeat</span>
                  <span :class="trainerMeta.heartbeat_healthy ? 'ok' : 'bad'">
                    <template v-if="trainerMeta.heartbeat_age_seconds != null">{{ trainerMeta.heartbeat_age_seconds }}s</template>
                    <template v-else>n/a</template>
                  </span>
                </p>
                <p v-if="trainerMeta.last_run"><span class="label">Start</span><span>{{ trainerMeta.last_run.start }}</span></p>
                <p v-if="trainerMeta.last_run"><span class="label">End</span><span>{{ trainerMeta.last_run.end }}</span></p>
                <p v-if="trainerMeta.last_run_age_seconds != null"><span class="label">Last Age</span><span>{{ trainerMeta.last_run_age_seconds }}s</span></p>
                <p v-if="trainerMeta.last_run?.base_days != null"><span class="label">Base Days</span><span>{{ trainerMeta.last_run.base_days }}</span></p>
                <p v-if="trainerMeta.last_run?.stacking_days != null"><span class="label">Stack Days</span><span>{{ trainerMeta.last_run.stacking_days }}</span></p>
              </div>
            </article>

            <article class="metric-card">
              <header>폴링 주기</header>
              <div class="metric-card__body interval-picker">
                <input type="range" min="5" max="45" step="5" v-model.number="pollInterval" />
                <span>{{ pollInterval }}초</span>
              </div>
            </article>
            <article class="metric-card" v-if="trainingStatus">
              <header>재학습 상태</header>
              <div class="metric-card__body meta-grid">
                <p><span class="label">Base Last</span><span>{{ trainingStatus.last_retrain_utc || '–' }}</span></p>
                <p><span class="label">Base Reason</span><span>{{ trainingStatus.last_retrain_reason || '–' }}</span></p>
                <p><span class="label">Meta Last</span><span>{{ trainingStatus.last_meta_retrain_utc || '–' }}</span></p>
                <p><span class="label">Meta Reason</span><span>{{ trainingStatus.last_meta_retrain_reason || '–' }}</span></p>
                <p><span class="label">Meta Overwritten</span><span>{{ trainingStatus.last_meta_retrain_overwritten === true ? 'Y' : (trainingStatus.last_meta_retrain_overwritten === false ? 'N' : '–') }}</span></p>
                <p><span class="label">Pending</span><span :class="trainingStatus.pending_meta_retrain ? 'warn' : 'ok'">{{ trainingStatus.pending_meta_retrain ? '대기' : '없음' }}</span></p>
                <p v-if="trainingStatus.meta_config?.every_minutes && trainingStatus.meta_config.every_minutes > 0"><span class="label">Meta Interval</span><span>{{ trainingStatus.meta_config.every_minutes }}m</span></p>
                <p v-else-if="trainingStatus.meta_config?.daily_at"><span class="label">Meta Daily</span><span>{{ trainingStatus.meta_config.daily_at }}</span></p>
                <p><span class="label">Min ΔBrier</span><span>{{ trainingStatus.meta_config?.min_rel_brier_improve ?? 0 }}</span></p>
                <p v-if="nextMetaEtaSeconds >= 0"><span class="label">Meta ETA</span><span>{{ formatEta(nextMetaEtaSeconds) }}</span></p>
                <p v-if="nextBaseEtaSeconds >= 0"><span class="label">Monthly ETA</span><span>{{ formatEta(nextBaseEtaSeconds) }}</span></p>
                <p><span class="label">Corr(Δ,samples)</span><span>{{ formatCorr(trainingStatus.meta_corr_rel_improve_samples) }}</span></p>
                <p><span class="label">Neg Streak</span><span :class="trainingStatus.meta_neg_streak_warn ? 'bad' : 'muted'">{{ trainingStatus.meta_neg_streak || 0 }}<template v-if="trainingStatus.meta_neg_streak_warn"> ⚠</template></span></p>
                <p><span class="label">Slope</span><span>{{ formatSlope(trainingStatus.meta_reg_slope) }}</span></p>
                <p><span class="label">p-value</span><span :class="pValueClass(trainingStatus.meta_reg_pvalue)">{{ formatPValue(trainingStatus.meta_reg_pvalue) }}</span></p>
                <p><span class="label">Coef Cos</span><span :class="coefCosClass(trainingStatus.meta_coef_cosine)">{{ formatCoefCos(trainingStatus.meta_coef_cosine) }}</span></p>
                <p><span class="label">Coef Drift</span><span :class="trainingStatus.meta_coef_drift_warn ? 'bad' : 'ok'">{{ trainingStatus.meta_coef_drift_warn ? 'WARN' : 'OK' }}</span></p>
                <div class="meta-sparkline-wrap">
                  <MetaHistorySpark
                    :history="trainingStatus.meta_history"
                    :warnNeg="trainingStatus.meta_neg_streak_warn"
                    :driftWarn="trainingStatus.meta_coef_drift_warn"
                  />
                </div>
              </div>
              <div class="metric-card__body meta-history" v-if="trainingStatus.meta_history && trainingStatus.meta_history.length">
                <p class="history-title">최근 메타 개선율</p>
                <div class="history-controls">
                  <label>
                    <span>표시 개수</span>
                    <input type="range" min="4" :max="trainingStatus.meta_history.length" v-model.number="historyDisplayCount" />
                    <strong>{{ historyDisplayCount }}</strong>
                  </label>
                  <button class="btn ghost" @click="exportMetaHistory">CSV 내보내기</button>
                </div>
                <div class="history-row">
                  <div v-for="h in displayedMetaHistory" :key="h.ts" class="hist-cell" :title="histTooltip(h)">
                    <span class="bar" :style="barStyleDynamic(h)"></span>
                    <span class="val">{{ histValue(h) }}</span>
                  </div>
                </div>
                <p class="history-title samples">샘플 수 추세</p>
                <div class="sample-spark">
                  <svg viewBox="0 0 120 28" preserveAspectRatio="none">
                    <polyline :points="sampleSparkPoints" />
                  </svg>
                  <div class="sample-labels">
                    <span v-for="h in displayedMetaHistory" :key="'s'+h.ts" class="samp" :title="'samples '+(h.samples||'n/a')">{{ shortSamples(h.samples) }}</span>
                  </div>
                </div>
                <p class="history-title scatter">개선율 vs 샘플</p>
                <div class="scatter-wrap">
                  <svg viewBox="0 0 120 60" preserveAspectRatio="none">
                    <g>
                      <rect x="0" y="0" width="120" height="60" fill="#111" stroke="#333" />
                      <polyline v-if="scatterFrame" :points="scatterFrame" stroke="#333" fill="none" />
                      <circle v-for="pt in scatterPoints" :key="pt.key" :cx="pt.x" :cy="pt.y" r="3" :fill="pt.color" :title="pt.tip" />
                    </g>
                  </svg>
                </div>
              </div>
            </article>
          </section>

          <NotificationCenter
            :items="notifications"
            @dismiss="dismissNotification"
          />
          <LayoutPresets
            :presets="layoutPresets"
            :currentFilter="symbolFilter"
            :sortKey="sortKey"
            :sortDir="sortDir"
            @save="savePreset"
            @apply="applyPreset"
            @delete="deletePreset"
          />
          <TradeSignalPanel
            :latest="latestTrade"
            :openTrades="openTrades"
            :activeSignals="activeSignals.length"
            :wsConnected="wsConnected"
          />
          <TradeHistoryList
            :trades="recentTradeHistory"
          />
          <ModelTrainingStatus
            v-if="modelTrainingRows.length"
            :items="modelTrainingRows"
          />
          <LowPriceIndicator
            :avgBottom="avgBottomScore"
            :activeShare="activeShare"
            :highShare="highConfidenceShare"
            :updatedAgo="lastUpdatedRelative"
            :wsConnected="wsConnected"
          />

          <main class="main-grid" v-if="displayedSymbols.length">
            <section class="symbol-board">
              <header class="board-header">
                <input class="filter-box" v-model="symbolFilter" placeholder="심볼 검색" />
                <div class="sort-controls">
                  <label>
                    <span>정렬</span>
                    <select v-model="sortKey">
                      <option value="stack_prob">스택 확률</option>
                      <option value="adj_prob">조정 확률</option>
                      <option value="bottom_score">바텀 점수</option>
                      <option value="symbol">심볼</option>
                      <option value="seq">SEQ 상태</option>
                    </select>
                  </label>
                  <label>
                    <span>방향</span>
                    <select v-model="sortDir">
                      <option value="desc">내림차순</option>
                      <option value="asc">오름차순</option>
                    </select>
                  </label>
                </div>
              </header>
              <div class="symbol-grid">
                <article
                  v-for="sym in sortedAndFilteredSymbols"
                  :key="sym"
                  :class="['symbol-card', sym === selectedSymbol ? 'selected' : '', nowcasts[sym].stacking?.decision ? 'decision-on' : 'decision-off']"
                  @click="selectedSymbol = sym"
                >
                  <header class="symbol-card__top">
                    <div>
                      <h2>{{ sym.toUpperCase() }}</h2>
                      <p class="muted">{{ formatInterval(nowcasts[sym].interval) }}</p>
                    </div>
                    <time :datetime="nowcasts[sym].timestamp">{{ formatTimestamp(nowcasts[sym].timestamp) }}</time>
                  </header>
                  <div class="symbol-card__metrics">
                    <div class="metric">
                      <span class="label">Price</span>
                      <strong>{{ formatPrice(nowcasts[sym].price) }}</strong>
                    </div>
                    <div class="metric">
                      <span class="label">Bottom</span>
                      <strong :class="scoreClass(nowcasts[sym].bottom_score)">{{ formatPct(nowcasts[sym].bottom_score) }}</strong>
                    </div>
                    <div class="metric" v-if="nowcasts[sym].stacking">
                      <span class="label">Stack</span>
                      <strong>{{ formatPct(nowcasts[sym].stacking?.prob) }}</strong>
                    </div>
                    <div class="metric" v-if="nowcasts[sym].stacking?.prob_final != null">
                      <span class="label">Adj</span>
                      <strong>{{ formatPct(nowcasts[sym].stacking?.prob_final) }}</strong>
                    </div>
                    <div class="metric" v-if="nowcasts[sym].stacking?.entry_meta">
                      <span class="label">Entry</span>
                      <strong :class="nowcasts[sym].stacking?.entry_meta?.entry_decision ? 'entry-on' : 'entry-off'">
                        {{ formatPct((nowcasts[sym].stacking?.entry_meta?.entry_prob ?? 0)) }}
                      </strong>
                    </div>
                    <div class="metric" v-if="nowcasts[sym].stacking">
                      <span class="label">Margin</span>
                      <strong :class="marginClass(nowcasts[sym].stacking?.margin)">{{ formatSignedPct(nowcasts[sym].stacking?.margin) }}</strong>
                    </div>
                  </div>
                  <div class="symbol-card__spark">
                    <svg viewBox="0 0 100 42" preserveAspectRatio="none">
                      <polyline :points="sparklinePoints(sym)" />
                    </svg>
                  </div>
                  <footer class="symbol-card__badges">
                    <span class="badge" :class="freshnessBadgeClass(sym)">{{ freshnessBadgeText(sym) }}</span>
                    <span class="badge" :class="gapBadgeClass(sym)">{{ gapBadgeText(sym) }}</span>
                    <span class="badge" :class="tfBadgeClass(sym)">{{ tfBadgeText(sym) }}</span>
                    <span class="badge" :class="seqBadgeClass(sym)">{{ seqBadgeText(sym) }}</span>
                    <span class="badge" :class="strengthClass(sym)">{{ shortStrength(sym) }}</span>
                    <span class="badge" :class="wrBadgeClass(sym)">{{ wrBadgeText(sym) }}</span>
                  </footer>
                </article>
              </div>
            </section>

            <section class="detail-panel" v-if="selectedSymbol && nowcasts[selectedSymbol]">
              <header class="detail-header">
                <div>
                  <h2>{{ selectedSymbol.toUpperCase() }} 상세</h2>
                  <p class="muted">{{ detailSubtitle }}</p>
                </div>
                <div class="pill-group" v-if="selectedStacking">
                  <span class="pill" :class="selectedStacking.decision ? 'on' : 'off'">
                    {{ selectedStacking.decision ? 'ON' : 'OFF' }}
                  </span>
                  <span class="pill muted">Conf {{ formatPct((selectedStacking.confidence ?? 0)) }}</span>
                  <span class="pill muted">Margin {{ formatSignedPct(selectedStacking.margin) }}</span>
                  <template v-if="Array.isArray(selectedStacking.used_models) && selectedStacking.used_models.length">
                    <span v-for="m in selectedStacking.used_models" :key="m" class="pill muted">{{ String(m).toUpperCase() }}</span>
                  </template>
                </div>
              </header>

              <section class="detail-grid">
                <article class="detail-card" v-if="modelStatusEntries.length">
                  <header>모델 준비도</header>
                  <ul class="list">
                    <li v-for="row in modelStatusEntries" :key="row.name">
                      <span>
                        <span class="status-dot" :class="row.ready ? 'ok' : 'bad'"></span>
                        {{ row.name.toUpperCase() }}
                      </span>
                      <span class="muted">
                        <template v-if="row.name === 'xgb'">
                          <span v-if="row.details.features != null">feat {{ row.details.features }}</span>
                          <span v-else>ready: {{ row.ready ? 'yes' : 'no' }}</span>
                        </template>
                        <template v-else>
                          <span v-if="row.details.have != null" :class="(row.details.need != null && row.details.have < row.details.need) ? 'need-missing' : ''">have {{ row.details.have }}</span>
                          <span v-if="row.details.seq_len != null"> · seq {{ row.details.seq_len }}</span>
                          <span v-if="row.details.feature_dim != null"> · dim {{ row.details.feature_dim }}</span>
                          <span v-if="row.details.padded"> · padded</span>
                          <span v-if="row.details.need != null"> · need {{ row.details.need }}</span>
                          <span v-if="!row.details.have && !row.details.seq_len && !row.details.feature_dim">ready: {{ row.ready ? 'yes' : 'no' }}</span>
                        </template>
                      </span>
                      <strong>{{ row.probText }}</strong>
                    </li>
                  </ul>
                </article>

                <article class="detail-card">
                  <header>확률 구조</header>
                  <ProbStructure v-if="selectedStacking" :stacking="selectedStacking" :threshold="selectedStacking.threshold" />
                  <MetaProbMetrics v-if="selectedStacking" :stacking="selectedStacking" />
                  <p v-else class="muted empty-note">Staking block unavailable.</p>
                </article>

                <article class="detail-card">
                  <header>기저 모델</header>
                  <ul class="list" v-if="baseProbEntries.length">
                    <li v-for="[model, value] in baseProbEntries" :key="model">
                      <span>{{ model }}</span>
                      <div class="progress compact">
                        <span :style="{ width: pctWidth(value) }"></span>
                      </div>
                      <strong>{{ formatBaseProb(value) }}</strong>
                    </li>
                  </ul>
                  <p v-else class="muted empty-note">모델 확률 데이터 없음</p>
                </article>

                <article class="detail-card">
                  <header>특징 기여도</header>
                  <ul class="list components">
                    <li v-for="(value, key) in filteredComponents(selectedNowcast?.components || {})" :key="key">
                      <span>{{ key }}</span>
                      <strong>{{ formatComponent(value) }}</strong>
                    </li>
                  </ul>
                </article>

                <article class="detail-card">
                  <header>데이터 품질</header>
                  <div class="quality" v-if="featureSnapshot">
                    <p><span class="label">Freshness</span><span>{{ freshnessBadgeText(selectedSymbol) }}</span></p>
                    <p><span class="label">누락 분</span><span>{{ featureSnapshot.missing_minutes_24h ?? 0 }}</span></p>
                    <p><span class="label">TF</span><span>{{ tfBadgeText(selectedSymbol) }}</span></p>
                    <p><span class="label">Source</span><span>{{ selectedNowcast?.price_source }}</span></p>
                  </div>
                  <p v-else class="muted empty-note">헬스 데이터를 기다리는 중…</p>
                </article>
              </section>
            </section>
          </main>

          <JobLogTable :items="jobLogs" />

          <section class="trades-pane" v-if="trades.length">
            <header class="pane-header">
              <h2>실거래 내역</h2>
              <button class="btn ghost" @click="manualTradesRefresh" :disabled="tradesLoading">새로고침</button>
            </header>
            <table class="trades-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>심볼</th>
                  <th>상태</th>
                  <th>레버리지</th>
                  <th>수량</th>
                  <th>진입</th>
                  <th>평단</th>
                  <th>현재</th>
                  <th>익절/손절</th>
                  <th>진행 손익</th>
                  <th>추가 매수</th>
                  <th>다음 추가</th>
                  <th>생성</th>
                  <th>종료</th>
                </tr>
              </thead>
              <tbody>
                <tr
                  v-for="row in tradeRows"
                  :key="row.kind === 'trade' ? `trade-${row.trade.id}` : `fills-${row.trade.id}`"
                  :class="[row.trade.status, row.kind === 'fills' ? 'fills-row' : '']"
                >
                  <template v-if="row.kind === 'trade'">
                    <td>
                      <div class="id-cell">
                        <span>{{ row.trade.id }}</span>
                        <button class="link" @click.prevent="toggleExpanded(row.trade.id)">
                          {{ expanded[row.trade.id] ? '접기' : '체결' }}
                        </button>
                      </div>
                    </td>
                    <td>{{ row.trade.symbol.toUpperCase() }}</td>
                    <td><span class="status-badge" :class="row.trade.status">{{ row.trade.status }}</span></td>
                    <td>{{ row.trade.leverage }}x</td>
                    <td>{{ row.trade.quantity.toFixed(3) }}</td>
                    <td>{{ formatNumber(row.trade.entry_price) }}</td>
                    <td>{{ formatNumber(row.trade.avg_price) }}</td>
                    <td>{{ row.trade.last_price != null ? formatNumber(row.trade.last_price) : '-' }}</td>
                    <td>{{ formatSignedPct(row.trade.take_profit_pct) }} / {{ formatSignedPct(row.trade.stop_loss_pct) }}</td>
                    <td :class="row.trade.pnl_pct_snapshot > 0 ? 'pos' : row.trade.pnl_pct_snapshot < 0 ? 'neg' : ''">
                      {{ formatSignedPct(row.trade.pnl_pct_snapshot) }}
                    </td>
                    <td>{{ row.trade.adds_done }}</td>
                    <td>
                      <span v-if="row.trade.next_add_in_seconds != null && row.trade.status === 'open'">
                        <span v-if="row.trade.next_add_in_seconds > 0">{{ formatDuration(row.trade.next_add_in_seconds) }}</span>
                        <span v-else class="pos">가능</span>
                      </span>
                      <span v-else>-</span>
                    </td>
                    <td>{{ formatShortDt(row.trade.created_at) }}</td>
                    <td>{{ row.trade.closed_at ? formatShortDt(row.trade.closed_at) : '-' }}</td>
                  </template>
                  <template v-else>
                    <td colspan="14">
                      <div class="fills-wrap">
                        <div class="fills-meta">
                          <span>마지막 체결 {{ row.trade.last_fill_at ? formatShortDt(row.trade.last_fill_at) : '-' }}</span>
                          <span v-if="row.trade.cooldown_seconds">쿨다운 {{ Math.floor(row.trade.cooldown_seconds / 60) }}분</span>
                        </div>
                        <table class="fills-table">
                          <thead>
                            <tr>
                              <th>시간</th>
                              <th>가격</th>
                              <th>수량</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr v-for="(fill, idx) in row.trade.fills" :key="idx">
                              <td>{{ formatShortDt(fill.t) }}</td>
                              <td class="num">{{ formatNumber(fill.price) }}</td>
                              <td class="num">{{ Number(fill.qty).toFixed(3) }}</td>
                            </tr>
                            <tr v-if="!row.trade.fills || !row.trade.fills.length">
                              <td colspan="3" class="muted">체결 기록 없음</td>
                            </tr>
                          </tbody>
                        </table>
                      </div>
                    </td>
                  </template>
                </tr>
              </tbody>
            </table>
          </section>

          <!-- Scheduler control panel (HTTP-only; appears if API reachable) -->
          <section class="system-strip">
            <SchedulerPanel
              :apiBase="apiBase"
              :trainingStatus="trainingStatus"
              :wsConnected="wsConnected"
              :useWs="true"
              :sendWs="sendAdminWs"
              :adminAck="adminAck"
            />
          </section>

          <section class="empty-state" v-if="!displayedSymbols.length">
            <p>나우캐스트 데이터를 기다리고 있습니다. 백엔드가 온라인인지 확인해주세요.</p>
          </section>
        </template>

        <section v-else class="schedule-pane">
          <header class="schedule-pane__header">
            <h2>스케줄 · 배치 제어</h2>
            <p class="muted">WS로 직접 제어하거나 HTTP 폴백을 사용할 수 있습니다.</p>
          </header>

          <div class="scheduler-controls">
            <div class="filters">
              <label>
                <span>타입</span>
                <select v-model="schedulerTypeFilter">
                  <option value="all">전체</option>
                  <option value="backend">Backend</option>
                  <option value="frontend">Frontend</option>
                </select>
              </label>
              <label>
                <span>상태</span>
                <select v-model="schedulerStatusFilter">
                  <option value="all">전체</option>
                  <option value="running">실행 중</option>
                  <option value="idle">대기</option>
                  <option value="disabled">비활성</option>
                </select>
              </label>
            </div>
            <div class="bulk-actions">
              <button class="btn ghost" @click="bulkSchedulerAction('stop_all')">전체 중지</button>
              <button class="btn ghost" @click="bulkSchedulerAction('start_all')">전체 재개</button>
            </div>
          </div>

          <div class="scheduler-status-grid" v-if="filteredSchedulerRows.length">
            <article v-for="row in filteredSchedulerRows" :key="row.key">
              <header>
                <strong>{{ row.name }}</strong>
                <span class="chip" :class="row.state">{{ stateLabel(row.state) }}</span>
              </header>
              <p class="muted">{{ row.desc || '설명 없음' }}</p>
              <div class="meta-line">
                <span>주기 {{ row.interval }}</span>
                <span>ETA {{ row.etaLabel }}</span>
              </div>
            </article>
          </div>

          <SchedulerList
            :items="filteredSchedulerRows"
            :loadingKey="schedulerCmdLoading"
            @command="handleSchedulerCommand"
          />
          <SchedulerPanel
            :apiBase="apiBase"
            :trainingStatus="trainingStatus"
            :wsConnected="wsConnected"
            :useWs="true"
            :sendWs="sendAdminWs"
            :adminAck="adminAck"
          />
          <SchedulerLog :items="schedulerLogs" />
        </section>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted, onBeforeUnmount } from 'vue'
import { formatPct, formatSignedPct, scoreClass, marginClass } from './utils'
// @ts-ignore script-setup SFC default export
import ProbStructure from './components/ProbStructure.vue'
// @ts-ignore script-setup SFC default export
import MetaProbMetrics from './components/MetaProbMetrics.vue'
// @ts-ignore script-setup SFC default export
import MetaHistorySpark from './components/MetaHistorySpark.vue'
// @ts-ignore script-setup SFC default export
import SchedulerPanel from './components/SchedulerPanel.vue'
// @ts-ignore script-setup SFC default export
import SchedulerList from './components/SchedulerList.vue'
// @ts-ignore script-setup SFC default export
import ModelTrainingStatus from './components/ModelTrainingStatus.vue'
// @ts-ignore script-setup SFC default export
import LowPriceIndicator from './components/LowPriceIndicator.vue'
// @ts-ignore script-setup SFC default export
import NotificationCenter from './components/NotificationCenter.vue'
// @ts-ignore script-setup SFC default export
import LayoutPresets from './components/LayoutPresets.vue'
// @ts-ignore script-setup SFC default export
import JobLogTable from './components/JobLogTable.vue'
// @ts-ignore script-setup SFC default export
import TradeSignalPanel from './components/TradeSignalPanel.vue'
// @ts-ignore script-setup SFC default export
import TradeHistoryList from './components/TradeHistoryList.vue'
// @ts-ignore script-setup SFC default export
import SchedulerLog from './components/SchedulerLog.vue'
// @ts-ignore script-setup SFC default export
import ChartView from './components/views/ChartView.vue'

interface NotificationItem {
  id: number
  ts: string
  level: 'info' | 'warn' | 'error'
  message: string
}
interface JobLogEntry {
  id: number
  ts: string
  source: string
  message: string
}
interface LayoutPreset {
  id: string
  name: string
  filter: string
  sortKey: typeof sortKey.value
  sortDir: typeof sortDir.value
}

interface ChartSignal {
  id: string | number
  ts: string
  side: 'buy' | 'sell'
  price?: number
  label?: string
}

type NowcastComponents = Record<string, number>
type NowcastEntry = Record<string, any>
type NowcastMap = Record<string, NowcastEntry>

const notifications = ref<NotificationItem[]>([])
const schedulerLogs = ref<Array<{ id: number; ts: string; message: string }>>([])
let schedulerLogSeq = 0
let notificationSeq = 0
let jobLogSeq = 0
const lastTradeFingerprint = ref<string>('')
const jobLogs = ref<JobLogEntry[]>([])
const presetStorageKey = 'rt-desk-layout-presets'
const layoutPresets = ref<LayoutPreset[]>(loadLayoutPresets())

const appendSchedulerLog = (message: string) => {
  const entry = { id: ++schedulerLogSeq, ts: new Date().toLocaleTimeString(), message }
  schedulerLogs.value = [entry, ...schedulerLogs.value].slice(0, 40)
}

const trainingStatus = ref<any | null>(null)
const trainingStatusUpdatedAt = ref<number>(Date.now())
const setTrainingStatus = (payload: any) => {
  if (!payload || typeof payload !== 'object') return
  trainingStatus.value = payload
  trainingStatusUpdatedAt.value = Date.now()
}

const schedulerTypeFilter = ref<'all' | 'backend' | 'frontend'>('all')
const schedulerStatusFilter = ref<'all' | 'running' | 'idle' | 'disabled'>('all')
const schedulerNowTick = ref<number>(Date.now())
let schedulerEtaTimer: number | null = null

const stateLabel = (state: string) => {
  if (state === 'running') return '실행 중'
  if (state === 'idle') return '대기'
  if (state === 'disabled') return '비활성'
  return state
}

const pushNotification = (message: string, level: NotificationItem['level'] = 'info') => {
  const entry = { id: ++notificationSeq, ts: new Date().toLocaleTimeString(), level, message }
  notifications.value = [entry, ...notifications.value].slice(0, 6)
}
const dismissNotification = (id: number) => {
  notifications.value = notifications.value.filter((item) => item.id !== id)
}
const addJobLog = (source: string, message: string) => {
  const entry = { id: ++jobLogSeq, ts: new Date().toLocaleTimeString(), source: source.toUpperCase(), message }
  jobLogs.value = [entry, ...jobLogs.value].slice(0, 30)
}
const persistPresets = () => {
  try {
    localStorage.setItem(presetStorageKey, JSON.stringify(layoutPresets.value))
  } catch {
    /* ignore */
  }
}
function loadLayoutPresets(): LayoutPreset[] {
  try {
    const raw = localStorage.getItem(presetStorageKey)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []
    return parsed.map((preset) => ({
      id: preset.id,
      name: preset.name || 'preset',
      filter: preset.filter || '',
      sortKey: preset.sortKey || 'stack_prob',
      sortDir: preset.sortDir || 'desc'
    }))
  } catch {
    return []
  }
}
const savePreset = (name: string) => {
  const trimmed = name.trim()
  if (!trimmed) return
  layoutPresets.value = [
    {
      id: `${Date.now().toString(36)}-${Math.random().toString(16).slice(2, 6)}`,
      name: trimmed,
      filter: symbolFilter.value,
      sortKey: sortKey.value,
      sortDir: sortDir.value
    },
    ...layoutPresets.value
  ].slice(0, 10)
  pushNotification(`프리셋 '${trimmed}' 저장 완료`)
}
const applyPreset = (id: string) => {
  const preset = layoutPresets.value.find((p) => p.id === id)
  if (!preset) return
  symbolFilter.value = preset.filter
  sortKey.value = preset.sortKey
  sortDir.value = preset.sortDir
  pushNotification(`프리셋 '${preset.name}' 적용`)
}
const deletePreset = (id: string) => {
  layoutPresets.value = layoutPresets.value.filter((p) => p.id !== id)
}

const openTrades = computed(() => trades.value.filter((t: any) => t.status === 'open'))
const latestTrade = computed(() => trades.value[0] || null)
const recentTradeHistory = computed(() => trades.value.slice(0, 8))

watch(latestTrade, (trade) => {
  if (!trade) return
  const fingerprint = `${trade.id}:${trade.status}:${trade.pnl_pct_snapshot ?? '0'}`
  if (fingerprint === lastTradeFingerprint.value) return
  lastTradeFingerprint.value = fingerprint
  pushNotification(
    `${trade.symbol?.toUpperCase() || 'UNKNOWN'} 상태 ${trade.status}`,
    trade.status === 'open' ? 'info' : trade.status === 'closed' ? 'warn' : 'info'
  )
  addJobLog('trade', `${trade.symbol} ${trade.status}`)
})

const nowcasts = ref<NowcastMap>({})
// Track per-symbol last update times (epoch ms) for diagnostics
const symbolLastUpdated = ref<Record<string, number>>({})
// Trainer 메타 (heartbeat + last run) 스냅샷
const trainerMeta = ref<Record<string, any> | null>(null)
const entryMetrics = ref<Record<string, any> | null>(null)
const loading = ref<boolean>(false)
const error = ref<string>('')
const pollInterval = ref<number>(10)
const lastUpdated = ref<Date | null>(null)
let pollTimer: number | null = null

const apiBase = ref<string>((window as any).VITE_API_BASE || 'http://127.0.0.1:8000')

// --- WebSocket support (no FastAPI HTTP) ---
let ws: WebSocket | null = null
const wsConnected = ref<boolean>(false)
const adminAck = ref<string>('')
let wsReconnectTimer: number | null = null
const wsTried = ref<string[]>([])
// 재연결 실패 횟수 및 연결 시작 시각
const wsFailCount = ref<number>(0)
const wsConnectedSince = ref<number | null>(null)
// @ts-ignore suppress import.meta module warning in certain TS service configs
// Avoid direct import.meta usage for environments where TS config disallows it; rely on global injection.
const forcedWs = (window as any).VITE_WS_URL || ''
const wsOnly = Boolean((window as any).WS_ONLY)

const wsCandidates = () => {
  if (forcedWs) return [forcedWs]
  const host = location.hostname || '127.0.0.1'
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  // Only target the dedicated WS backend port to avoid noisy failures.
  return [`${proto}://${host}:8022`]
}

const handleWsMessage = (evt: MessageEvent) => {
  try {
    const msg = JSON.parse(evt.data)
    if (msg?.type === 'snapshot') {
      if (msg.nowcast && typeof msg.nowcast === 'object') nowcasts.value = msg.nowcast
      if (msg.features && typeof msg.features === 'object') features.value = msg.features
      if (Array.isArray(msg.trades)) trades.value = msg.trades
      if (msg.training_status && typeof msg.training_status === 'object') setTrainingStatus(msg.training_status)
      if (msg.trainer && typeof msg.trainer === 'object') trainerMeta.value = msg.trainer
      if (msg.entry_metrics && typeof msg.entry_metrics === 'object') entryMetrics.value = msg.entry_metrics
      lastUpdated.value = new Date()
      try {
        Object.keys(nowcasts.value).forEach((sym) => { symbolLastUpdated.value[sym] = Date.now() })
      } catch {}
      // Stop HTTP polling if snapshot received
      stopPolling()
      // Also stop trades HTTP polling when WS provides snapshot
      stopTradesPolling()
      // Stop features HTTP polling when WS provides snapshot
      stopFeaturesPolling()
      // Stop training status HTTP polling when WS provides snapshot
      stopTrainingStatusPolling()
    } else if (msg?.type === 'nowcast' && msg.symbol && msg.data) {
      nowcasts.value = { ...nowcasts.value, [String(msg.symbol)]: msg.data }
      lastUpdated.value = new Date()
      symbolLastUpdated.value[String(msg.symbol)] = Date.now()
      try {
        console.debug('[ws] nowcast', msg.symbol, 'price=', msg.data.price, 'bottom=', msg.data.bottom_score)
      } catch {}
    } else if (msg?.type === 'trades' && Array.isArray(msg.data)) {
      trades.value = msg.data
      // Receiving trades via WS; ensure HTTP trades polling is off
      stopTradesPolling()
    } else if (msg?.type === 'features' && typeof msg.data === 'object') {
      features.value = { ...features.value, ...msg.data }
      // Receiving features via WS; ensure HTTP features polling is off
      stopFeaturesPolling()
    } else if (msg?.type === 'training_status' && msg.data && typeof msg.data === 'object') {
      setTrainingStatus(msg.data)
    } else if (msg?.type === 'trainer' && msg.data && typeof msg.data === 'object') {
      trainerMeta.value = msg.data
    } else if (msg?.type === 'entry_metrics' && msg.data && typeof msg.data === 'object') {
      entryMetrics.value = msg.data
    } else if (msg?.type === 'admin_ack') {
      try {
        const ok = !!msg.ok
        const action = msg.action || 'cmd'
        if (ok) adminAck.value = `WS: ${action} 완료`
        else adminAck.value = `WS: ${action} 실패${msg.error ? ' - ' + msg.error : ''}`
      } catch {}
    }
  } catch {
    /* ignore */
  }
}

const connectWs = async () => {
  if (ws && ws.readyState === WebSocket.OPEN) return
  const candidates = wsCandidates()
  for (const url of candidates) {
    if (wsTried.value.includes(url)) continue
    wsTried.value.push(url)
    try {
      console.debug('[ws] attempt', url)
      const sock = new WebSocket(url)
      await new Promise<void>((resolve, reject) => {
        const to = window.setTimeout(() => reject(new Error('timeout')), 2200)
        sock.onopen = () => { window.clearTimeout(to); resolve() }
        sock.onerror = () => { window.clearTimeout(to); reject(new Error('ws error')) }
      })
      ws = sock
      wsConnected.value = true
      wsFailCount.value = 0
      wsConnectedSince.value = Date.now()
      console.info('[ws] connected', url)
      ws.onmessage = handleWsMessage
      ws.onclose = () => {
        console.warn('[ws] closed', url)
        wsConnected.value = false
        wsFailCount.value += 1
        // Fallback to HTTP polling while reconnecting
        startPolling()
        startFeaturesPolling()
        startTrainingStatusPolling()
        scheduleWsReconnect()
        adminAck.value = ''
      }
      stopPolling()
      // Stop HTTP trades polling once WS is connected
      stopTradesPolling()
      // Stop HTTP features polling once WS is connected
      stopFeaturesPolling()
      // Stop HTTP training status polling once WS is connected
      stopTrainingStatusPolling()
      return
    } catch (err) {
      console.warn('[ws] failed', url, (err as any)?.message)
    }
  }
  wsConnected.value = false
  console.warn('[ws] all candidates failed; reverting to HTTP polling')
  startPolling()
  startTrainingStatusPolling()
}

const scheduleWsReconnect = () => {
  if (wsReconnectTimer) window.clearTimeout(wsReconnectTimer)
  // 지수 백오프: 4s * 2^failCount (최대 60s)
  const base = 4000
  const delay = Math.min(base * Math.pow(2, wsFailCount.value), 60000)
  wsReconnectTimer = window.setTimeout(connectWs, delay)
}

// Periodic retry while in HTTP fallback mode
let wsRetryInterval: number | null = null
const startWsPeriodicRetry = () => {
  if (wsRetryInterval) return
  wsRetryInterval = window.setInterval(() => {
    if (!wsConnected.value) connectWs()
    else if (wsRetryInterval) { window.clearInterval(wsRetryInterval); wsRetryInterval = null }
  }, 20000)
}
const stopWsPeriodicRetry = () => {
  if (wsRetryInterval) { window.clearInterval(wsRetryInterval); wsRetryInterval = null }
}

const features = ref<Record<string, any>>({})
let featuresTimer: number | null = null
const trades = ref<any[]>([])
const tradesLoading = ref<boolean>(false)
const expanded = ref<Record<number, boolean>>({})
let tradesTimer: number | null = null
let trainingStatusTimer: number | null = null

const nextMetaEtaSeconds = computed(() => {
  const id = 'meta_retrain_interval'
  const dailyId = 'meta_retrain_daily'
  const src = trainingStatus.value?.next_runs?.[id] || trainingStatus.value?.next_runs?.[dailyId]
  if (!src || src.eta_seconds == null) return -1
  return Number(src.eta_seconds)
})
const nextBaseEtaSeconds = computed(() => {
  const monthly = trainingStatus.value?.next_runs?.['monthly_train']
  if (!monthly || monthly.eta_seconds == null) return -1
  return Number(monthly.eta_seconds)
})
function formatEta(sec: number) {
  if (sec < 0) return '–'
  if (sec < 120) return `${sec}s`
  const m = Math.round(sec / 60)
  if (m < 120) return `${m}m`
  const h = Math.round(m / 60)
  if (h < 48) return `${h}h`
  const d = Math.round(h / 24)
  return `${d}d`
}
function sendAdminWs(action: string) {
  try {
    const token = (window as any).WS_ADMIN_TOKEN || (window as any).ADMIN_TOKEN || ''
    if (!ws || ws.readyState !== WebSocket.OPEN) throw new Error('ws not connected')
    ws.send(JSON.stringify({ type: 'admin', action, token }))
    adminAck.value = `WS: ${action} 전송`
    addJobLog('admin', `전송 ${action}`)
  } catch (e: any) {
    adminAck.value = `WS 전송 실패: ${e?.message || 'error'}`
  }
}
function histTooltip(h: any) {
  if (!h) return ''
  return `${h.ts}\nΔBrier=${h.rel_improve != null ? (h.rel_improve*100).toFixed(2)+'%' : 'n/a'}\nBrier_new=${h.brier_new}`
}
function barStyle(h: any) {
  const v = h?.rel_improve
  if (v == null || isNaN(v)) return { height: '4px', background: '#444', width: '100%' }
  const pct = Math.min(100, Math.max(0, v * 100))
  const hue = v > 0 ? 140 : 0
  return { height: '4px', background: `hsl(${hue} 70% 40%)`, width: '100%', opacity: 0.85 }
}
function histValue(h: any) {
  const v = h?.rel_improve
  if (v == null || isNaN(v)) return '–'
  return (v*100).toFixed(1)+'%'
}
// History display controls
const historyDisplayCount = ref<number>(8)
const displayedMetaHistory = computed<any[]>(() => {
  const hist: any[] = trainingStatus.value?.meta_history || []
  if (!hist.length) return []
  const n = Math.max(1, Math.min(historyDisplayCount.value, hist.length))
  return hist.slice(-n)
})
function barStyleDynamic(h: any) {
  const v = h?.rel_improve
  if (v == null || isNaN(v)) return { height: '4px', background: '#444', width: '100%' }
  // Scale saturation & lightness by magnitude
  const mag = Math.min(1, Math.abs(v) * 10) // if rel_improve 0.1 -> mag=1
  const hue = v > 0 ? 140 : 0
  const sat = 40 + mag * 40  // 40%..80%
  const light = 30 + mag * 15 // 30%..45%
  return { height: '4px', background: `hsl(${hue} ${sat}% ${light}%)`, width: '100%', transition: 'background 0.3s' }
}
// Scatter plot (improvement vs samples)
const scatterPoints = computed<any[]>(() => {
  const hist: any[] = displayedMetaHistory.value
  if (!hist.length) return []
  const imps = hist.map(h => (typeof h.rel_improve === 'number' ? h.rel_improve : null)).filter(v => v != null) as number[]
  const samples = hist.map(h => (typeof h.samples === 'number' ? h.samples : null)).filter(v => v != null) as number[]
  if (!imps.length || !samples.length) return []
  const maxImp = Math.max(...imps.map(i => Math.abs(i)), 0.0001)
  const maxSamples = Math.max(...samples, 1)
  return hist.map((h, idx) => {
    const imp = typeof h.rel_improve === 'number' ? h.rel_improve : 0
    const samp = typeof h.samples === 'number' ? h.samples : 0
    // X: normalized improvement magnitude, sign encoded by side (pos right, neg left of center)
    const w = 120; const hgt = 60
    const xNorm = (Math.abs(imp) / maxImp)
    const x = (imp >= 0 ? 60 + xNorm * 60 : 60 - xNorm * 60)
    const y = hgt - (samp / maxSamples) * (hgt - 6) - 3
    const hue = imp >= 0 ? 140 : 0
    const sat = 50 + Math.min(50, Math.abs(imp) * 400) // scale saturation
    const light = 40
    return { x, y, key: String(h.ts)+'_'+idx, color: `hsl(${hue} ${sat}% ${light}%)`, tip: `imp=${(imp*100).toFixed(2)}% samples=${samp}` }
  })
})
const scatterFrame = computed(() => '0,0 120,0 120,60 0,60 0,0')
const sampleSparkPoints = computed((): string => {
  const hist: any[] = trainingStatus.value?.meta_history || []
  if (!hist.length) return ''
  const samples: number[] = hist.map((h: any) => typeof h.samples === 'number' ? h.samples : 0)
  const max = Math.max(...samples, 1)
  const w = 120; const hgt = 24
  return samples.map((s: number, i: number) => {
    const x = (i / Math.max(1, samples.length - 1)) * w
    const y = hgt - (s / max) * hgt
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
})
function shortSamples(n: any) {
  if (typeof n !== 'number' || isNaN(n)) return '–'
  if (n >= 1000) return (n/1000).toFixed(1)+'k'
  return String(n)
}
function formatCorr(c: any) {
  if (c == null || isNaN(Number(c))) return 'n/a'
  return Number(c).toFixed(3)
}
function formatSlope(s: any) {
  if (s == null || isNaN(Number(s))) return 'n/a'
  // slope is rel_improve per sample → show scaled (% per 1k samples) if small
  const val = Number(s)
  const scaled = val * 100 // convert improvement fraction to % per sample
  if (Math.abs(scaled) < 0.001) return (scaled * 1000).toFixed(4)+'%/1k'
  return scaled.toFixed(4)+'%/sample'
}
function formatPValue(p: any) {
  if (p == null || isNaN(Number(p))) return 'n/a'
  const v = Number(p)
  if (v < 0.0001) return '<1e-4'
  return v.toFixed(4)
}
function pValueClass(p: any) {
  if (p == null || isNaN(Number(p))) return 'muted'
  const v = Number(p)
  if (v <= 0.05) return 'ok'
  if (v <= 0.15) return 'warn'
  return 'muted'
}
function formatCoefCos(c: any) {
  if (c == null || isNaN(Number(c))) return 'n/a'
  return Number(c).toFixed(4)
}
function coefCosClass(c: any) {
  if (c == null || isNaN(Number(c))) return 'muted'
  const v = Number(c)
  if (v >= 0.97) return 'ok'
  if (v >= 0.92) return 'warn'
  return 'bad'
}
const exportMetaHistory = () => {
  try {
    const hist: any[] = trainingStatus.value?.meta_history || []
    if (!hist.length) return
    const header = ['ts','brier_old','brier_new','rel_improve','overwritten','samples']
    const rows = hist.map(h => [h.ts, h.brier_old, h.brier_new, h.rel_improve, h.overwritten, h.samples])
    const csv = [header.join(','), ...rows.map(r => r.map(v => v == null ? '' : String(v)).join(','))].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const a = document.createElement('a')
    const url = URL.createObjectURL(blob)
    a.href = url
    a.download = 'meta_retrain_history.csv'
    a.style.display = 'none'
    document.body.appendChild(a)
    a.click()
    setTimeout(() => { URL.revokeObjectURL(url); a.remove() }, 400)
  } catch (e) {
    console.warn('export failed', e)
  }
}

const symbolFilter = ref<string>('')
const sortKey = ref<'stack_prob' | 'adj_prob' | 'bottom_score' | 'symbol' | 'seq'>('stack_prob')
const sortDir = ref<'asc' | 'desc'>('desc')
const selectedSymbol = ref<string>('')

const displayedSymbols = computed<string[]>(() =>
  Object.keys(nowcasts.value)
    .filter((key) => !key.startsWith('_') && !!nowcasts.value[key])
)

const sortedAndFilteredSymbols = computed<string[]>(() => {
  let list = displayedSymbols.value
  if (symbolFilter.value.trim()) {
    const q = symbolFilter.value.trim().toLowerCase()
    list = list.filter((sym) => sym.toLowerCase().includes(q))
  }

  const sorted = [...list]
  sorted.sort((a, b) => {
    if (sortKey.value === 'symbol') {
      const cmp = a.localeCompare(b)
      return sortDir.value === 'asc' ? cmp : -cmp
    }
    const getValue = (sym: string) => {
      const nc = nowcasts.value[sym]
      if (!nc) return 0
      if (sortKey.value === 'bottom_score') return Number(nc.bottom_score ?? 0)
      if (sortKey.value === 'seq') return seqReadinessScore(sym)
      if (sortKey.value === 'adj_prob') return Number(nc.stacking?.prob_final ?? nc.stacking?.prob ?? 0)
      return Number(nc.stacking?.prob ?? 0)
    }
    const diff = getValue(a) - getValue(b)
    return sortDir.value === 'asc' ? diff : -diff
  })
  return sorted
})

watch(displayedSymbols, (syms) => {
  if (!selectedSymbol.value && syms.length) {
    selectedSymbol.value = syms[0]
  } else if (selectedSymbol.value && !syms.includes(selectedSymbol.value) && syms.length) {
    selectedSymbol.value = syms[0]
  } else if (!syms.length) {
    selectedSymbol.value = ''
  }
})

const chartSymbol = ref<string>('')

watch(displayedSymbols, (syms) => {
  if (!syms.length) {
    chartSymbol.value = ''
    return
  }
  if (!chartSymbol.value || !syms.includes(chartSymbol.value)) {
    chartSymbol.value = syms[0]
  }
})

const chartNowcast = computed(() => (chartSymbol.value ? nowcasts.value[chartSymbol.value] ?? null : null))
const chartSignals = computed<ChartSignal[]>(() => {
  const sym = chartSymbol.value
  if (!sym) return []
  const nc = nowcasts.value[sym]
  if (!nc) return []
  const signals = (nc.signals || nc.chart_signals) as ChartSignal[] | undefined
  if (!Array.isArray(signals)) return []
  return signals
})

const stackingMeta = computed(() => (nowcasts.value._stacking_meta as { method?: string; method_override?: string | null; threshold?: number | null; threshold_source?: string | null; used_models?: string[]; models?: string[]; entry?: any }) || null)
const entryMetaCard = computed(() => {
  const m = entryMetrics.value || {}
  const th = (nowcasts.value._stacking_meta && (nowcasts.value._stacking_meta as any).entry) || {}
  return {
    win_rate: typeof (m as any)?.overall?.win_rate === 'number' ? (m as any).overall.win_rate : null,
    samples: typeof (m as any)?.overall?.samples === 'number' ? (m as any).overall.samples : 0,
    window: typeof (m as any)?.window === 'number' ? (m as any).window : 0,
    target: typeof (m as any)?.target_effective === 'number' ? (m as any).target_effective : (typeof (m as any)?.target === 'number' ? (m as any).target : null),
    th_env: typeof (m as any)?.threshold?.env === 'number' ? (m as any).threshold.env : (typeof (th as any)?.threshold_env === 'number' ? (th as any).threshold_env : null),
    th_sidecar: typeof (m as any)?.threshold?.sidecar === 'number' ? (m as any).threshold.sidecar : (typeof (th as any)?.threshold_sidecar === 'number' ? (th as any).threshold_sidecar : null),
    th_dynamic: typeof (m as any)?.threshold?.dynamic === 'number' ? (m as any).threshold.dynamic : (typeof (th as any)?.threshold_dynamic === 'number' ? (th as any).threshold_dynamic : null),
  }
})
const entryWrSparkPoints = computed(() => {
  const hist = (entryMetrics.value?.history_overall || []) as Array<{ ts: string; wr?: number | null }>
  if (!hist.length) return ''
  const vals = hist.map(h => (typeof h.wr === 'number' ? Math.max(0, Math.min(1, h.wr)) : null)).filter(v => v != null) as number[]
  if (!vals.length) return ''
  const w = 120, h = 30
  return vals.map((v, i) => {
    const x = (i / Math.max(1, vals.length - 1)) * w
    const y = h - v * h
    return `${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
})

const activeSignals = computed(() =>
  displayedSymbols.value.filter((sym) => nowcasts.value[sym]?.stacking?.decision)
)

const highConfidenceSignals = computed(() =>
  displayedSymbols.value.filter((sym) => {
    const conf = Number(nowcasts.value[sym]?.stacking?.confidence ?? 0)
    return conf >= 0.2
  })
)

const activeShare = computed(() => {
  if (!displayedSymbols.value.length) return 0
  return Math.round((activeSignals.value.length / displayedSymbols.value.length) * 100)
})

const highConfidenceShare = computed(() => {
  if (!displayedSymbols.value.length) return 0
  return Math.round((highConfidenceSignals.value.length / displayedSymbols.value.length) * 100)
})

const avgBottomScore = computed(() => {
  const values = displayedSymbols.value
    .map((sym) => nowcasts.value[sym]?.bottom_score)
    .filter((value): value is number => typeof value === 'number' && !Number.isNaN(value))
  if (!values.length) return null
  const sum = values.reduce((acc, value) => acc + value, 0)
  return sum / values.length
})

watch(layoutPresets, persistPresets, { deep: true })
watch(wsConnected, (next, prev) => {
  if (prev === undefined) return
  pushNotification(next ? 'WS 연결' : 'WS 분리 · HTTP 폴백', next ? 'info' : 'warn')
  addJobLog('ws', next ? 'connected' : 'disconnected')
})
watch(() => trainingStatus.value?.pending_meta_retrain, (next, prev) => {
  if (next && !prev) pushNotification('메타 재학습 대기 중', 'warn')
})
watch(avgBottomScore, (next, prev) => {
  if (next == null) return
  if ((prev ?? 0) < 0.65 && next >= 0.65) pushNotification('평균 바텀 점수가 경보 구간 돌입', 'warn')
})

const lastUpdatedRelative = computed(() => {
  if (!lastUpdated.value) return '미동기화'
  const diffMs = Date.now() - lastUpdated.value.getTime()
  const seconds = Math.max(0, Math.round(diffMs / 1000))
  if (seconds < 60) return `${seconds}초 전`
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes}분 전`
  const hours = Math.round(minutes / 60)
  return `${hours}시간 전`
})

const healthDot = computed(() => {
  if (error.value) return 'bad'
  if (!displayedSymbols.value.length) return 'warn'
  if (activeSignals.value.length) return 'ok'
  return 'warn'
})

const healthText = computed(() => {
  if (error.value) return `에러: ${error.value}`
  if (!displayedSymbols.value.length) return '데이터 없음 · 백엔드 확인 필요'
  if (activeSignals.value.length) return '정상 작동 중 · 신호 감지'
  return '정상 작동 중 · 대기'
})

// 연결 메트릭 (UI에서 추후 활용 가능)
const connectionUptimeSeconds = computed(() => {
  if (!wsConnectedSince.value || !wsConnected.value) return 0
  return Math.round((Date.now() - wsConnectedSince.value) / 1000)
})
const reconnectAttempts = computed(() => wsFailCount.value)

const selectedNowcast = computed(() => (selectedSymbol.value ? nowcasts.value[selectedSymbol.value] : null))
const selectedStacking = computed(() => selectedNowcast.value?.stacking && selectedNowcast.value.stacking.ready ? selectedNowcast.value.stacking : null)
const baseProbEntries = computed<[string, number | null][]>(() => {
  if (!selectedNowcast.value?.base_probs) return []
  const rows: Array<[string, number | null]> = []
  for (const [k, v] of Object.entries(selectedNowcast.value.base_probs)) {
    // Preserve null/undefined so formatter can show 'n/a' instead of 0
    rows.push([k.toUpperCase(), v == null ? null : Number(v)])
  }
  return rows
})

const modelStatusEntries = computed(() => {
  const out: Array<{ name: string; ready: boolean; details: any; probText: string }> = []
  const nc = selectedNowcast.value
  if (!nc) return out
  const probs = (nc.base_probs || {}) as Record<string, number | null>
  const info = (nc.base_info || {}) as Record<string, any>
  for (const name of Object.keys({ ...probs, ...info })) {
    if (name === 'base_logits') continue
    const ready = Boolean(info?.[name]?.ready ?? (probs[name] != null))
    const details = info?.[name] || {}
    const p = probs[name]
    const probText = p == null || Number.isNaN(Number(p)) ? 'n/a' : formatBaseProb(Number(p))
    out.push({ name, ready, details, probText })
  }
  // stable order: xgb, lstm, tf, then others alpha
  out.sort((a, b) => {
    const order = ['xgb', 'lstm', 'tf']
    const ia = order.indexOf(a.name)
    const ib = order.indexOf(b.name)
    if (ia !== -1 || ib !== -1) return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
    return a.name.localeCompare(b.name)
  })
  return out
})

const modelTrainingRows = computed(() => {
  const rows = trainingStatus.value?.model_training_rows || trainingStatus.value?.model_training
  if (Array.isArray(rows)) return rows
  return []
})

const featureSnapshot = computed(() => (selectedSymbol.value ? features.value[selectedSymbol.value] : null))

const detailSubtitle = computed(() => {
  if (!selectedNowcast.value) return ''
  return `${formatTimestamp(selectedNowcast.value.timestamp)} · ${selectedNowcast.value.price_source}`
})

const pctWidth = (value?: number | null) => {
  if (value == null || isNaN(Number(value))) return '0%'
  const pct = Math.max(0, Math.min(1, Number(value)))
  return `${(pct * 100).toFixed(0)}%`
}

const sparklinePoints = (sym: string) => {
  const comp = nowcasts.value[sym]?.components || {}
  const values = Object.values(comp).map((v) => Number(v)).filter((v) => !Number.isNaN(v))
  if (!values.length) return '0,21 100,21'
  const slice = values.slice(0, 18)
  const max = Math.max(...slice)
  const min = Math.min(...slice)
  const range = max - min || 1
  return slice
    .map((val, idx) => {
      const x = slice.length === 1 ? 100 : (idx / (slice.length - 1)) * 100
      const norm = (val - min) / range
      const y = 36 - norm * 32
      return `${x.toFixed(2)},${y.toFixed(2)}`
    })
    .join(' ')
}

const shortStrength = (sym: string) => {
  const cls = strengthClass(sym)
  if (cls === 'strength-high') return '강'
  if (cls === 'strength-mid') return '중'
  return '약'
}

const formatThreshold = (value?: number | null) => {
  if (value == null || Number.isNaN(Number(value))) return 'auto'
  return Number(value).toFixed(2)
}

const formatTimestamp = (ts: string) => {
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

const formatInterval = (interval: string) => interval?.toUpperCase?.() || interval

const formatPrice = (price: number) => {
  if (Math.abs(price) >= 100) return price.toFixed(2)
  return price.toFixed(4)
}

const formatNumber = (value: number) => {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  if (Math.abs(value) >= 100) return Number(value).toFixed(2)
  if (Math.abs(value) >= 1) return Number(value).toFixed(3)
  return Number(value).toFixed(4)
}

const formatDuration = (seconds: number) => {
  const sec = Math.max(0, Math.floor(seconds))
  const minutes = Math.floor(sec / 60)
  const rest = sec % 60
  return `${minutes}:${String(rest).padStart(2, '0')}`
}

const formatShortDt = (value: string) => {
  try {
    return new Date(value).toLocaleString()
  } catch {
    return value
  }
}

const formatComponent = (value: number) => {
  if (Math.abs(value) >= 1000) return value.toFixed(0)
  if (Math.abs(value) >= 10) return value.toFixed(2)
  return value.toFixed(4)
}

const formatBaseProb = (value?: number | null) => {
  if (value == null || Number.isNaN(Number(value))) return 'n/a'
  const percent = Number(value) * 100
  const abs = Math.abs(percent)
  if (abs >= 1) return `${percent.toFixed(1)}%`
  if (abs >= 0.1) return `${percent.toFixed(2)}%`
  if (abs >= 0.01) return `${percent.toFixed(3)}%`
  return `${percent.toFixed(4)}%`
}

const filteredComponents = (components: NowcastComponents) => {
  const hidden = ['logit']
  const items: Record<string, number> = {}
  Object.entries(components || {}).forEach(([key, value]) => {
    if (!hidden.includes(key)) items[key] = value
  })
  return items
}

const strengthClass = (sym: string) => {
  const stack = nowcasts.value[sym]?.stacking
  if (!stack?.ready) return 'strength-low'
  let score = 0
  const margin = Number(stack.margin ?? 0)
  const conf = Number(stack.confidence ?? 0)
  const bottom = Number(nowcasts.value[sym]?.bottom_score ?? 0)
  if (margin >= 0.05) score += 2
  else if (margin >= 0.02) score += 1
  if (conf >= 0.2) score += 2
  else if (conf >= 0.1) score += 1
  if (bottom >= 0.7) score += 2
  else if (bottom >= 0.55) score += 1
  if (score >= 5) return 'strength-high'
  if (score >= 3) return 'strength-mid'
  return 'strength-low'
}

const freshnessBadgeClass = (sym: string) => {
  const snapshot = features.value[sym]
  if (!snapshot) return 'stale'
  const sec = Number(snapshot.data_fresh_seconds ?? 999)
  if (sec < 90) return 'fresh'
  if (sec < 300) return 'stale'
  return 'cold'
}

const freshnessBadgeText = (sym: string) => {
  const cls = freshnessBadgeClass(sym)
  if (cls === 'fresh') return 'Fresh'
  if (cls === 'cold') return 'Cold'
  return 'Stale'
}

// Relative age (seconds) for symbol's last nowcast tick
const symbolAgeSeconds = (sym: string) => {
  const t = symbolLastUpdated.value[sym]
  if (!t) return null
  return Math.round((Date.now() - t) / 1000)
}

const gapBadgeClass = (sym: string) => {
  const snapshot = features.value[sym]
  const missing = Number(snapshot?.missing_minutes_24h ?? 0)
  if (missing === 0) return 'no-gaps'
  if (missing <= 10) return 'minor-gaps'
  return 'gaps'
}

const gapBadgeText = (sym: string) => {
  const snapshot = features.value[sym]
  const missing = Number(snapshot?.missing_minutes_24h ?? 0)
  return missing === 0 ? 'No Gaps' : `Gaps:${missing}`
}

const tfBadgeClass = (sym: string) => {
  const snapshot = features.value[sym]
  if (!snapshot) return 'tf-missing'
  const has5 = Boolean(snapshot['5m_latest_open_time'])
  const has15 = Boolean(snapshot['15m_latest_open_time'])
  if (has5 && has15) return 'tf-ok'
  if (has5 || has15) return 'tf-partial'
  return 'tf-missing'
}

const tfBadgeText = (sym: string) => {
  const cls = tfBadgeClass(sym)
  if (cls === 'tf-ok') return 'TF OK'
  if (cls === 'tf-partial') return 'TF Partial'
  return 'TF Missing'
}

// Per-symbol WR badge based on entryMetrics.by_symbol
const wrForSymbol = (sym: string) => {
  const m: any = entryMetrics.value || {}
  const row = m.by_symbol?.[sym]
  if (!row) return { wr: null as number | null, samples: 0, target: (m.target as number | null) ?? null }
  const wr = typeof row.win_rate === 'number' ? row.win_rate : null
  const samples = typeof row.samples === 'number' ? row.samples : 0
  const target = typeof m.target === 'number' ? m.target : null
  return { wr, samples, target }
}
const wrBadgeClass = (sym: string) => {
  const { wr, samples, target } = wrForSymbol(sym)
  if (wr == null || samples <= 0 || target == null) return 'muted'
  if (wr >= target) return 'wr-good'
  if (wr >= target - 0.05) return 'wr-mid'
  return 'wr-bad'
}
const wrBadgeText = (sym: string) => {
  const { wr } = wrForSymbol(sym)
  if (wr == null) return 'WR –'
  return `WR ${Math.round(wr * 100)}%`
}

// SEQ readiness badge based on base_info have/need for lstm/tf
const seqBadgeClass = (sym: string) => {
  const nc = nowcasts.value[sym]
  const bi = nc?.base_info || {}
  const lj = bi['lstm'] || {}
  const tj = bi['tf'] || {}
  const haveL = Number(lj.have ?? 0)
  const needL = Number(lj.need ?? 0)
  const haveT = Number(tj.have ?? 0)
  const needT = Number(tj.need ?? 0)
  const okL = needL ? haveL >= needL : Boolean(lj.ready)
  const okT = needT ? haveT >= needT : Boolean(tj.ready)
  if (okL && okT) return 'fresh'
  if (okL || okT) return 'minor-gaps'
  return 'gaps'
}
const seqBadgeText = (sym: string) => {
  const cls = seqBadgeClass(sym)
  if (cls === 'fresh') return 'SEQ OK'
  if (cls === 'minor-gaps') return 'SEQ Partial'
  return 'SEQ Missing'
}

// Numeric score for seq readiness to support sorting (OK=2, Partial=1, Missing=0)
const seqReadinessScore = (sym: string) => {
  const cls = seqBadgeClass(sym)
  if (cls === 'fresh') return 2
  if (cls === 'minor-gaps') return 1
  return 0
}

const toggleExpanded = (id: number) => {
  expanded.value[id] = !expanded.value[id]
}

const tradeRows = computed(() => {
  const rows: Array<{ kind: 'trade' | 'fills'; trade: any }> = []
  trades.value.forEach((trade) => {
    rows.push({ kind: 'trade', trade })
    if (expanded.value[trade.id]) rows.push({ kind: 'fills', trade })
  })
  return rows
})

const fetchNowcast = async () => {
  loading.value = true
  error.value = ''
  try {
    const response = await fetch(`${apiBase.value}/nowcast`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const ct = response.headers.get('content-type') || ''
    if (!ct.toLowerCase().includes('application/json')) {
      throw new Error('non-json response')
    }
    const data = await response.json()
    if (data && typeof data === 'object' && !Array.isArray(data)) {
      if (data.symbol && Object.keys(data).length <= 12) {
        const sym = String(data.symbol).toLowerCase()
        nowcasts.value = { [sym]: data }
      } else {
        nowcasts.value = data as NowcastMap
      }
      lastUpdated.value = new Date()
    } else {
      nowcasts.value = {}
    }
  } catch (err: any) {
    error.value = err?.message || '데이터를 불러오는 데 실패했습니다'
  } finally {
    loading.value = false
  }
}

const fetchFeaturesForSymbol = async (sym: string) => {
  try {
    const response = await fetch(`${apiBase.value}/health/features?symbol=${encodeURIComponent(sym)}`)
    if (!response.ok) return
    const data = await response.json()
    features.value = { ...features.value, [sym]: data }
  } catch {
    /* swallow */
  }
}

const fetchTrades = async () => {
  tradesLoading.value = true
  try {
    const response = await fetch(`${apiBase.value}/trades?limit=50`)
    if (!response.ok) return
    const ct = response.headers.get('content-type') || ''
    if (!ct.toLowerCase().includes('application/json')) return
    const data = await response.json()
    if (Array.isArray(data)) trades.value = data
  } catch {
    /* swallow */
  } finally {
    tradesLoading.value = false
  }
}
const fetchTrainingStatus = async () => {
  if (!apiBase.value) return
  try {
    const response = await fetch(`${apiBase.value}/training/status`)
    if (!response.ok) return
    const data = await response.json()
    setTrainingStatus(data)
    if (data?.trainer) trainerMeta.value = data.trainer
    if (data?.entry_metrics) entryMetrics.value = data.entry_metrics
  } catch {
    /* swallow */
  }
}
const manualRefresh = async () => {
  await Promise.all([fetchNowcast(), fetchTrades(), fetchTrainingStatus()])
}

const manualTradesRefresh = () => fetchTrades()

const stopPolling = () => {
  if (pollTimer) {
    window.clearInterval(pollTimer)
    pollTimer = null
  }
}

const startPolling = () => {
  stopPolling()
  pollTimer = window.setInterval(fetchNowcast, pollInterval.value * 1000)
}

const stopFeaturesPolling = () => {
  if (featuresTimer) {
    window.clearInterval(featuresTimer)
    featuresTimer = null
  }
}

const startFeaturesPolling = () => {
  stopFeaturesPolling()
  const runner = () => displayedSymbols.value.forEach((sym) => fetchFeaturesForSymbol(sym))
  runner()
  featuresTimer = window.setInterval(runner, 45_000)
}

const stopTradesPolling = () => {
  if (tradesTimer) {
    window.clearInterval(tradesTimer)
    tradesTimer = null
  }
}

const stopTrainingStatusPolling = () => {
  if (trainingStatusTimer) {
    window.clearInterval(trainingStatusTimer)
    trainingStatusTimer = null
  }
}
const startTradesPolling = () => {
  stopTradesPolling()
  const runner = () => fetchTrades()
  runner()
  tradesTimer = window.setInterval(runner, 45_000)
}
const startTrainingStatusPolling = () => {
  stopTrainingStatusPolling()
  const runner = () => fetchTrainingStatus()
  runner()
  trainingStatusTimer = window.setInterval(runner, 60_000)
}

const autoDetectApiBase = async () => {
  if ((window as any).VITE_API_BASE) return
  // WS-first: avoid probing HTTP on 8022 (WS server)
  const candidates: string[] = ['http://127.0.0.1:8000']
  const sameOrigin = `${location.protocol}//${location.host}`
  if (!/:5173\b/.test(location.host)) candidates.unshift(sameOrigin)

  const tryHealth = async (base: string) => {
    const controller = new AbortController()
    const timeout = window.setTimeout(() => controller.abort(), 1200)
    try {
      const response = await fetch(`${base}/health`, {
        signal: controller.signal,
        headers: { Accept: 'application/json' }
      })
      if (!response.ok) return false
      const type = response.headers.get('content-type') || ''
      if (!type.toLowerCase().includes('application/json')) return false
      const payload = await response.json().catch(() => null)
      return payload && payload.status === 'ok'
    } catch {
      return false
    } finally {
      window.clearTimeout(timeout)
    }
  }

  for (const candidate of candidates) {
    const ok = await tryHealth(candidate)
    if (ok) {
      apiBase.value = candidate
      ;(window as any).VITE_API_BASE = candidate
      break
    }
  }
}

watch(pollInterval, () => {
  startPolling()
})

const sideMenuItems = [
  { key: 'chart', label: '차트' },
  { key: 'monitoring', label: '모니터링' },
  { key: 'schedule', label: '스케줄/배치' }
] as const
type ViewKey = (typeof sideMenuItems)[number]['key']
const activeView = ref<ViewKey>('monitoring')
const setView = (view: ViewKey) => { activeView.value = view }

const schedulerCmdLoading = ref<string>('')
const formatSchedulerInterval = (seconds?: number) => {
  if (typeof seconds !== 'number' || Number.isNaN(seconds) || seconds <= 0) return 'n/a'
  if (seconds % 3600 === 0) return `${Math.round(seconds / 3600)}h`
  if (seconds % 60 === 0) return `${Math.round(seconds / 60)}m`
  return `${seconds}s`
}
const schedulerRows = computed(() => {
  const base = Array.isArray(trainingStatus.value?.schedulers)
    ? trainingStatus.value?.schedulers
    : Object.entries(trainingStatus.value?.next_runs || {}).map(([key, meta]: [string, any]) => ({
        key,
        name: meta?.label || key,
        desc: meta?.description || '',
        interval: meta?.every_minutes ? `${meta.every_minutes}m` : formatSchedulerInterval(meta?.every_seconds),
        lastRun: meta?.last_run || meta?.last_run_at || null,
        running: meta?.running === true,
        enabled: meta?.enabled !== false,
        tags: meta?.tags || meta?.models || [],
        type: meta?.scope || 'backend'
      }))
  return (base || []).map((row: any) => {
    const meta = trainingStatus.value?.next_runs?.[row.key] || {}
    const etaBase = typeof meta.eta_seconds === 'number' ? meta.eta_seconds : -1
    const delta = (schedulerNowTick.value - trainingStatusUpdatedAt.value) / 1000
    const etaSeconds = etaBase >= 0 ? Math.max(0, etaBase - delta) : -1
    return {
      ...row,
      etaSeconds,
      etaLabel: etaSeconds >= 0 ? formatEta(Math.round(etaSeconds)) : '–',
      state: row.running ? 'running' : row.enabled ? 'idle' : 'disabled'
    }
  })
})
const filteredSchedulerRows = computed(() =>
  schedulerRows.value.filter((row: any) => {
    const typeOk = schedulerTypeFilter.value === 'all' || row.type === schedulerTypeFilter.value
    const statusOk = schedulerStatusFilter.value === 'all' || row.state === schedulerStatusFilter.value
    return typeOk && statusOk
  })
)

const bulkSchedulerAction = (mode: 'stop_all' | 'start_all') => {
  appendSchedulerLog(mode === 'stop_all' ? '전체 중지 명령 전송' : '전체 재개 명령 전송')
  sendAdminWs(`scheduler:${mode}`)
}

const handleSchedulerCommand = ({ key, action }: { key: string; action: 'start' | 'stop' }) => {
  schedulerCmdLoading.value = `${key}:${action}`
  addJobLog('scheduler', `${action.toUpperCase()} ${key}`)
  appendSchedulerLog(`${action.toUpperCase()} ${key}`)
  try {
    sendAdminWs(`scheduler:${action}:${key}`)
  } finally {
    window.setTimeout(() => (schedulerCmdLoading.value = ''), 600)
  }
}

const startSchedulerEtaTimer = () => {
  if (schedulerEtaTimer) return
  schedulerEtaTimer = window.setInterval(() => {
    schedulerNowTick.value = Date.now()
  }, 1000)
}
const stopSchedulerEtaTimer = () => {
  if (!schedulerEtaTimer) return
  window.clearInterval(schedulerEtaTimer)
  schedulerEtaTimer = null
}

onMounted(async () => {
  await connectWs()
  if (!wsConnected.value && !wsOnly) {
    await autoDetectApiBase()
    await Promise.all([fetchNowcast(), fetchTrades(), fetchTrainingStatus()])
    startPolling()
    startWsPeriodicRetry()
    startFeaturesPolling()
    startTradesPolling()
    startTrainingStatusPolling()
  }
  startSchedulerEtaTimer()
})

onBeforeUnmount(() => {
  stopPolling()
  stopFeaturesPolling()
  stopTradesPolling()
  stopTrainingStatusPolling()
  stopWsPeriodicRetry()
  stopSchedulerEtaTimer()
  if (ws) try { ws.close() } catch {}
})
</script>

<style scoped>
:root {
  --bg: #0b1220;
  --card: linear-gradient(185deg, rgba(22, 30, 54, 0.95), rgba(13, 19, 36, 0.92));
  --card-alt: linear-gradient(180deg, rgba(21, 30, 52, 0.94), rgba(12, 18, 34, 0.9));
  --border: rgba(65, 90, 135, 0.45);
  --text: #e3e9ff;
  --muted: #8ea0c4;
  --ok: #4ade80;
  --warn: #fbbf24;
  --bad: #f87171;
  --accent: #60a5fa;
  --glow-1: rgba(96, 165, 250, 0.25);
  --glow-2: rgba(59, 130, 246,  0.2);
}

body {
  background: radial-gradient(circle at 20% 20%, rgba(96, 152, 255, 0.16), transparent 60%),
    radial-gradient(circle at 80% 0%, rgba(59, 130, 246, 0.18), transparent 65%),
    linear-gradient(160deg, #05070f, var(--bg));
  color: var(--text);
}

.app-shell {
  color: var(--text);
  background: linear-gradient(170deg, rgba(8, 12, 24, 0.9), rgba(13, 19, 34, 0.82));
}

.app-shell::before {
  background: radial-gradient(circle, rgba(96, 165, 250, 0.35), transparent 72%);
}

.app-shell::after {
  background: radial-gradient(circle, rgba(14, 165, 233, 0.28), transparent 74%);
}

.app-shell > * {
  position: relative;
  z-index: 1;
}

.api-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.45rem 0.75rem;
  background: rgba(255, 255, 255, 0.75);
  border: 1px solid rgba(129, 161, 214, 0.4);
  border-radius: 9999px;
  font-size: 0.85rem;
  color: var(--muted);
  box-shadow: 0 8px 18px -12px rgba(60, 100, 180, 0.4);
}

.api-input {
  width: 220px;
  border: none;
  background: transparent;
  color: var(--text);
  font-weight: 600;
  font-size: 0.85rem;
}

.api-input:focus {
  outline: none;
}

.status-chip {
  padding: 0.4rem 0.8rem;
  border-radius: 9999px;
  font-size: 0.78rem;
  border: 1px solid transparent;
}

.status-chip.ok {
  background: rgba(74, 222, 128, 0.15);
  border-color: rgba(74, 222, 128, 0.4);
}

.status-chip.warn {
  background: rgba(251, 191, 36, 0.15);
  border-color: rgba(251, 191, 36, 0.45);
}

.status-chip.bad {
  background: rgba(248, 113, 113, 0.18);
  border-color: rgba(248, 113, 113, 0.45);
}

.system-strip {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  margin-bottom: 1.5rem;
}

.metric-card {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 20px;
  background: var(--card);
  box-shadow: 0 40px 80px -55px rgba(5, 8, 15, 0.9);
  display: flex;
  flex-direction: column;
  backdrop-filter: saturate(140%) blur(12px);
}

.metric-card header {
  padding: 0.85rem 1rem 0.35rem;
  font-size: 0.85rem;
  letter-spacing: 0.8px;
  color: var(--muted);
  text-transform: uppercase;
}

.metric-card__body {
  padding: 0 1rem 1rem;
  display: grid;
  gap: 0.4rem;
}

.health-line {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0;
  font-weight: 600;
}

.dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: var(--muted);
}

.dot.ok {
  background: var(--ok);
}

.dot.warn {
  background: var(--warn);
}

.dot.bad {
  background: var(--bad);
}

.muted {
  color: var(--muted);
  font-size: 0.85rem;
}

.meta-grid p {
  margin: 0;
  display: flex;
  justify-content: space-between;
  font-size: 0.85rem;
}

.meta-grid .label {
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.models-line {
  word-break: break-all;
}

.distribution {
  display: grid;
  gap: 0.6rem;
}

.dist-row {
  display: grid;
  grid-template-columns: 60px 1fr auto;
  gap: 0.75rem;
  align-items: center;
  font-size: 0.8rem;
}

.bar {
  position: relative;
  height: 7px;
  background: rgba(255, 255, 255, 0.08);
  border-radius: 999px;
  overflow: hidden;
}

.bar span {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(34, 197, 94, 0.1), rgba(34, 197, 94, 0.6));
}

.bar.warn span {
  background: linear-gradient(90deg, rgba(250, 204, 21, 0.1), rgba(250, 204, 21, 0.6));
}

.interval-picker {
  display: grid;
  gap: 0.5rem;
}

.interval-picker input[type='range'] {
  width: 100%;
}

.main-grid {
  display: grid;
  gap: 1.5rem;
  grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
  align-items: start;
}

.symbol-board {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 24px;
  background: var(--card-alt);
  padding: 1.35rem 1.45rem 1.6rem;
  backdrop-filter: saturate(150%) blur(16px);
  box-shadow: 0 32px 60px -34px rgba(67, 108, 186, 0.32);
  position: relative;
  overflow: hidden;
}

.symbol-board::before {
  content: '';
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 50% -20%, rgba(173, 205, 255, 0.55), transparent 62%);
  opacity: 0.5;
}

.symbol-board::after {
  content: '';
  position: absolute;
  inset: 0;
  background-image: linear-gradient(rgba(208, 219, 255, 0.18) 1px, transparent 1px),
    linear-gradient(90deg, rgba(208, 219, 255, 0.18) 1px, transparent 1px);
  background-size: 52px 52px;
  opacity: 0.28;
  mask-image: radial-gradient(circle, rgba(255, 255, 255, 0.85), transparent 70%);
}

.symbol-board > * {
  position: relative;
  z-index: 1;
}

.board-header {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
}

.filter-box {
  flex: 1;
  min-width: 180px;
  padding: 0.6rem 0.8rem;
  border-radius: 12px;
  border: 1px solid rgba(129, 161, 214, 0.35);
  background: rgba(255, 255, 255, 0.78);
  color: var(--text);
  box-shadow: 0 12px 26px -18px rgba(67, 108, 186, 0.4);
}

.sort-controls {
  display: flex;
  gap: 0.75rem;
  font-size: 0.8rem;
  color: var(--muted);
}

.sort-controls label {
  display: flex;
  gap: 0.4rem;
  align-items: center;
}

.sort-controls select {
  padding: 0.35rem 0.6rem;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: rgba(255, 255, 255, 0.85);
  color: var(--text);
  box-shadow: 0 12px 20px -16px rgba(67, 108, 186, 0.4);
}

.symbol-grid {
  display: grid;
  gap: 0.8rem;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
}

.symbol-card {
  border: 1px solid rgba(129, 161, 214, 0.28);
  border-radius: 18px;
  padding: 0.95rem;
  cursor: pointer;
  transition: transform 0.28s ease, border-color 0.25s ease, box-shadow 0.28s ease;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.92), rgba(235, 243, 255, 0.94));
  box-shadow: 0 24px 44px -32px rgba(67, 108, 186, 0.38);
}

.symbol-card:hover {
  transform: translateY(-6px);
  border-color: rgba(36, 107, 255, 0.42);
  box-shadow: 0 32px 56px -30px rgba(36, 107, 255, 0.42);
}

.symbol-card.selected {
  transform: translateY(-4px);
  border-color: rgba(36, 107, 255, 0.48);
  box-shadow: 0 28px 52px -32px rgba(36, 107, 255, 0.45);
}

.symbol-card.decision-on {
  border-color: rgba(34, 197, 94, 0.36);
}

.symbol-card__top {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
}

.symbol-card__top h2 {
  margin: 0;
  font-size: 1.1rem;
}

.symbol-card__top time {
  font-size: 0.72rem;
  color: var(--muted);
}

.symbol-card__metrics {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.6rem;
  margin: 0.8rem 0 0.6rem;
}

.symbol-card__metrics .label {
  display: block;
  color: var(--muted);
  font-size: 0.75rem;
}

.symbol-card__metrics strong {
  display: block;
  font-size: 0.95rem;
}
.symbol-card__metrics strong.entry-on {
  color: var(--ok);
  text-shadow: 0 0 4px rgba(34,197,94,0.3);
}
.symbol-card__metrics strong.entry-off {
  color: var(--muted);
  opacity: 0.7;
}

.symbol-card__spark {
  height: 46px;
}

.symbol-card__spark svg {
  width: 100%;
  height: 100%;
  stroke: var(--accent);
  stroke-width: 1.8;
  fill: none;
  opacity: 0.9;
}

.symbol-card__badges {
  display: flex;
  gap: 0.4rem;
  flex-wrap: wrap;
  margin-top: 0.6rem;
}

.badge {
  display: inline-flex;
  padding: 0.25rem 0.5rem;
  border-radius: 9999px;
  font-size: 0.68rem;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  background: rgba(59, 72, 104, 0.35);
  color: var(--muted);
}

.badge.fresh {
  background: rgba(34, 197, 94, 0.2);
  color: var(--ok);
}

.badge.cold {
  background: rgba(248, 113, 113, 0.18);
  color: var(--bad);
}

.badge.no-gaps {
  background: rgba(34, 197, 94, 0.16);
  color: var(--ok);
}

.badge.minor-gaps {
  background: rgba(250, 204, 21, 0.2);
  color: var(--warn);
}

.badge.gaps {
  background: rgba(248, 113, 113, 0.16);
  color: var(--bad);
}

.badge.tf-ok {
  background: rgba(96, 165, 250, 0.2);
  color: #bfdbfe;
}

.badge.tf-partial {
  background: rgba(129, 140, 248, 0.22);
  color: #c7d2fe;
}


.badge.tf-missing {
  background: rgba(148, 163, 184, 0.28);
  color: var(--muted);
}

.badge.strength-high {
  background: rgba(16, 185, 129, 0.24);
  color: #6ee7b7;
}

.badge.strength-mid {
  background: rgba(250, 204, 21, 0.25);
  color: #fde68a;
}

.badge.strength-low {
  background: rgba(148, 163, 184, 0.24);
  color: var(--muted);
}

/* Win-rate badge styles */
.badge.wr-good {
  background: rgba(34, 197, 94, 0.2);
  color: var(--ok);
}
.badge.wr-mid {
  background: rgba(250, 204, 21, 0.2);
  color: var(--warn);
}
.badge.wr-bad {
  background: rgba(248, 113, 113, 0.18);
  color: var(--bad);
}

.wr-spark {
  width: 100%;
  height: 40px;
}
.wr-spark polyline {
  stroke: var(--accent);
  stroke-width: 2;
  fill: none;
  opacity: 0.85;
}

.detail-panel {
  border: 1px solid rgba(129, 161, 214, 0.32);
  border-radius: 24px;
  background: var(--card);
  padding: 1.6rem 1.5rem;
  display: grid;
  gap: 1.25rem;
  box-shadow: 0 36px 68px -38px rgba(67, 108, 186, 0.35);
  backdrop-filter: saturate(150%) blur(18px);
}

.detail-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.detail-header h2 {
  margin: 0;
  font-size: 1.2rem;
}

.pill-group {
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
}

.pill {
  padding: 0.35rem 0.7rem;
  border-radius: 999px;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.4px;
}

.pill.on {
  background: rgba(34, 197, 94, 0.25);
  color: var(--ok);
}

.pill.off {
  background: rgba(148, 163, 184, 0.2);
  color: var(--muted);
}

.pill.muted {
  background: rgba(255, 255, 255, 0.6);
  color: var(--muted);
}

.detail-grid {
  display: grid;
  gap: 1rem;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
}

.detail-card {
  border: 1px solid rgba(129, 161, 214, 0.28);
  border-radius: 18px;
  background: linear-gradient(185deg, rgba(255, 255, 255, 0.92), rgba(233, 241, 255, 0.94));
  padding: 1rem 1.2rem;
  display: grid;
  gap: 0.85rem;
  box-shadow: 0 18px 36px -26px rgba(67, 108, 186, 0.28);
}

.detail-card header {
  font-size: 0.85rem;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  color: var(--muted);
}

.prob-bars {
  display: grid;
  gap: 0.6rem;
}

.bar-row {
  display: grid;
  grid-template-columns: 60px 1fr auto;
  gap: 0.75rem;
  align-items: center;
  font-size: 0.8rem;
}

.progress {
  position: relative;
  height: 8px;
  background: rgba(129, 161, 214, 0.18);
  border-radius: 999px;
  overflow: hidden;
}

.progress span {
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, rgba(96, 165, 250, 0.2), rgba(96, 165, 250, 0.7));
}

.progress span.smooth {
  background: linear-gradient(90deg, rgba(129, 140, 248, 0.25), rgba(129, 140, 248, 0.7));
}

.progress span.final {
  background: linear-gradient(90deg, rgba(14, 165, 233, 0.3), rgba(14, 165, 233, 0.8));
}

.progress span.threshold {
  background: linear-gradient(90deg, rgba(245, 158, 11, 0.3), rgba(245, 158, 11, 0.7));
}

.progress.compact {
  height: 6px;
}

.list {
  list-style: none;
  margin: 0;
  padding: 0;
  display: grid;
  gap: 0.55rem;
  font-size: 0.82rem;
}

.list li {
  display: grid;
  grid-template-columns: 1fr minmax(0, 120px) auto;
  gap: 0.6rem;
  align-items: center;
}

.components li {
  grid-template-columns: 1fr auto;
}

.quality {
  display: grid;
  gap: 0.4rem;
  font-size: 0.82rem;
}

.quality .label {
  color: var(--muted);
  margin-right: 0.6rem;
}

.empty-note {
  margin: 0;
}

.app-layout {
  display: flex;
  gap: 1.5rem;
  align-items: flex-start;
}
.side-menu {
  background: linear-gradient(180deg, rgba(9, 13, 26, 0.95), rgba(15, 23, 42, 0.92));
  color: var(--text);
  border-color: rgba(99, 121, 173, 0.4);
}
.side-menu__logo h1 {
  margin: 0;
  font-size: 1.05rem;
  letter-spacing: 0.6px;
}
.side-menu__logo p {
  margin: 0.15rem 0 0;
  font-size: 0.78rem;
  opacity: 0.8;
}
.side-menu__title {
  margin: 1.2rem 0 0.6rem;
  font-size: 0.78rem;
  letter-spacing: 0.4px;
  text-transform: uppercase;
  color: var(--muted);
}
.side-menu__nav {
  display: grid;
  gap: 0.6rem;
  margin-top: 1.4rem;
}
.side-menu__btn {
  border: 1px solid rgba(96, 112, 150, 0.35);
  background: rgba(15, 23, 42, 0.4);
  color: inherit;
}
.side-menu__btn.active {
  background: rgba(37, 99, 235, 0.35);
  border-color: rgba(96, 165, 250, 0.7);
}
.placeholder-card,
.schedule-pane {
  background: linear-gradient(185deg, rgba(18, 26, 48, 0.95), rgba(10, 15, 28, 0.92));
  border-color: rgba(81, 110, 163, 0.45);
  color: var(--text);
}
.metric-card,
.symbol-board,
.detail-panel,
.detail-card,
.trades-pane {
  background: var(--card);
  border-color: rgba(81, 110, 163, 0.45);
  box-shadow: 0 40px 80px -55px rgba(5, 8, 15, 0.9);
}
.symbol-card {
  background: var(--card-alt);
  border-color: rgba(81, 110, 163, 0.4);
}
.filter-box,
.sort-controls select,
.api-pill {
  background: rgba(15, 23, 42, 0.8);
  color: var(--text);
  border-color: rgba(91, 118, 170, 0.45);
}
.status-chip.ok {
  background: rgba(74, 222, 128, 0.15);
  border-color: rgba(74, 222, 128, 0.4);
}
.status-chip.warn {
  background: rgba(251, 191, 36, 0.15);
  border-color: rgba(251, 191, 36, 0.45);
}
.status-chip.bad {
  background: rgba(248, 113, 113, 0.18);
  border-color: rgba(248, 113, 113, 0.45);
}
.btn.ghost {
  border-color: rgba(148, 163, 184, 0.4);
  color: var(--text);
}
.trades-table th,
.trades-table td,
.fills-table th,
.fills-table td {
  border-color: rgba(99, 121, 173, 0.25);
}

.scheduler-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 1rem;
  justify-content: space-between;
  margin-bottom: 1rem;
}
.scheduler-controls label {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  font-size: 0.78rem;
}
.scheduler-controls select {
  padding: 0.35rem 0.6rem;
  border-radius: 10px;
  border: 1px solid rgba(99, 121, 173, 0.4);
  background: rgba(15, 23, 42, 0.75);
  color: var(--text);
}
.scheduler-status-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.9rem;
  margin-bottom: 1.2rem;
}
.scheduler-status-grid article {
  border: 1px solid rgba(129, 161, 214, 0.3);
  border-radius: 18px;
  padding: 0.8rem 1rem;
  background: rgba(9, 14, 28, 0.85);
}
.scheduler-status-grid header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.4rem;
}
.chip {
  padding: 0.2rem 0.55rem;
  border-radius: 999px;
  font-size: 0.72rem;
  text-transform: uppercase;
  border: 1px solid rgba(148, 163, 184, 0.4);
}
.chip.running {
  border-color: rgba(34, 197, 94, 0.4);
  color: var(--ok);
}
.chip.idle {
  border-color: rgba(251, 191, 36, 0.4);
  color: var(--warn);
}
.chip.disabled {
  border-color: rgba(148, 163, 184, 0.4);
  color: var(--muted);
}
.scheduler-status-grid .meta-line {
  display: flex;
  justify-content: space-between;
  font-size: 0.78rem;
  color: var(--muted);
}

</style>
