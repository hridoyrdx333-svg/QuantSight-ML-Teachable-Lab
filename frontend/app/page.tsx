 'use client'

import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { useEffect, useState } from 'react'
import {
  Activity, BarChart3, BrainCircuit, CircleHelp, Database, Gauge,
  LayoutDashboard, LineChart, LockKeyhole, Menu, Radio, Settings,
  ShieldCheck, SlidersHorizontal, Sparkles, Target, WalletCards, X,
  ChevronDown, Clock3, ServerCog, ShieldAlert, RefreshCcw, FileText,
  Boxes, CircleDollarSign, TimerReset, ListChecks, TrendingUp, Zap,
  HardDrive, BookOpenCheck, Binary, DatabaseZap, Waypoints
} from 'lucide-react'

const navItems = [
  ['Dashboard','/dashboard',LayoutDashboard],
  ['Market','/market',LineChart],
  ['AI Model','/ai-model',BrainCircuit],
  ['Signals','/signals',Target],
  ['Paper Trading','/paper-trading',WalletCards],
  ['Outcomes','/outcomes',BarChart3],
  ['Risk','/risk',ShieldCheck],
  ['System Health','/system-health',Gauge],
  ['Data & Evidence','/data-evidence',Database],
  ['Settings','/settings',Settings],
] as const

const markets = [
  {symbol:'BTCUSDT', price:'Awaiting API', change:'—', funding:'—', basis:'—', regime:'Awaiting API', fresh:'Pending'},
  {symbol:'ETHUSDT', price:'Awaiting API', change:'—', funding:'—', basis:'—', regime:'Awaiting API', fresh:'Pending'},
  {symbol:'SOLUSDT', price:'Awaiting API', change:'—', funding:'—', basis:'—', regime:'Awaiting API', fresh:'Pending'},
]

const signals = [
  ['BTCUSDT','Awaiting API','0.56','—','—','Pending','OFF'],
  ['ETHUSDT','Awaiting API','0.56','—','—','Pending','OFF'],
  ['SOLUSDT','Awaiting API','0.54','—','—','Pending','OFF'],
]

function Badge({children,tone='neutral'}:{children:React.ReactNode;tone?:string}) {
  return <span className={`badge ${tone}`}>{children}</span>
}

function Stat({
  label,value,detail,tone='cyan',icon
}:{
  label:string; value:string; detail:string; tone?:string; icon?:React.ReactNode
}) {
  return <article className="stat-card">
    <div className="stat-head"><span>{label}</span>{icon ?? <span className={`tiny-dot ${tone}`} />}</div>
    <strong>{value}</strong>
    <small className={`${tone}-text`}>{detail}</small>
  </article>
}

function Panel({
  eyebrow,title,children,action,className=''
}:{
  eyebrow:string; title:string; children:React.ReactNode; action?:React.ReactNode; className?:string
}) {
  return <article className={`panel ${className}`}>
    <div className="panel-head">
      <div><p className="eyebrow">{eyebrow}</p><h3>{title}</h3></div>
      {action}
    </div>
    {children}
  </article>
}

function StatusCard({
  icon,label,value,detail,tone='cyan'
}:{
  icon:React.ReactNode; label:string; value:string; detail:string; tone?:string
}) {
  return <div className="health-card">
    <div className={`health-icon ${tone}`}>{icon}</div>
    <div className="health-copy">
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </div>
    <Badge tone={tone}>{value}</Badge>
  </div>
}

function MetricStrip({items}:{items:{label:string;value:string;hint:string;tone?:string}[]}) {
  return <div className="metric-strip">
    {items.map((item)=><div key={item.label}>
      <span>{item.label}</span>
      <strong className={item.tone?`${item.tone}-text`:''}>{item.value}</strong>
      <small>{item.hint}</small>
    </div>)}
  </div>
}

function Table({headers, rows}:{headers:string[];rows:string[][]}) {
  return <div className="data-table">
    <div className="table-row table-header" style={{gridTemplateColumns:`repeat(${headers.length}, minmax(0,1fr))`}}>
      {headers.map(h=><span key={h}>{h}</span>)}
    </div>
    {rows.map((row,i)=><div className="table-row" key={i} style={{gridTemplateColumns:`repeat(${headers.length}, minmax(0,1fr))`}}>
      {row.map((cell,j)=><span key={j}>{cell}</span>)}
    </div>)}
  </div>
}

function PlaceholderChart(){
  return <div className="spark-area">
    <svg viewBox="0 0 600 150" preserveAspectRatio="none" aria-hidden="true">
      <path d="M0,122 C60,122 72,110 110,108 C160,104 160,83 205,86 C248,89 260,58 305,68 C352,78 366,40 413,51 C459,62 472,32 520,39 C559,44 575,20 600,26" fill="none" stroke="currentColor" strokeWidth="3"/>
      <path d="M0,122 C60,122 72,110 110,108 C160,104 160,83 205,86 C248,89 260,58 305,68 C352,78 366,40 413,51 C459,62 472,32 520,39 C559,44 575,20 600,26 L600,150 L0,150 Z" fill="currentColor" opacity=".05"/>
    </svg>
    <span>Visualization ready for live API data</span>
  </div>
}

function Hero({eyebrow,title,sub,action}:{eyebrow:string;title:string;sub:string;action?:React.ReactNode}) {
  return <section className="hero-row">
    <div><p className="eyebrow">{eyebrow}</p><h2>{title}</h2><p className="subtle">{sub}</p></div>
    {action}
  </section>
}


type QuantStatus = any

function prob(v:any){
  const n = Number(v)
  return Number.isFinite(n) ? `${(n*100).toFixed(2)}%` : "—"
}

function age(v:any){
  const n = Number(v)
  if(!Number.isFinite(n)) return "—"
  if(n < 60) return `${Math.round(n)} sec`
  if(n < 3600) return `${Math.round(n/60)} min`
  return `${(n/3600).toFixed(1)} hr`
}

function liveRows(status:QuantStatus | null){
  const latest = status?.shadow?.latest ?? {}
  const thresholds = status?.shadow?.status?.thresholds ?? {}

  return ["BTCUSDT","ETHUSDT","SOLUSDT"].map(symbol=>{
    const x = latest[symbol] ?? {}

    return [
      symbol,
      prob(x.probability_up),
      String(
        x.locked_threshold ??
        thresholds[symbol] ??
        (symbol==="SOLUSDT" ? 0.54 : 0.56)
      ),
      String(x.decision ?? "—"),
      age(x.feature_age_seconds),
      x.orders_sent ? "SENT" : "OFF"
    ]
  })
}

function MarketPage(){
  return <>
    <Hero eyebrow="MARKET CONTEXT / READ-ONLY FEED" title="Tracked instruments" sub="Live market context slots for BTC, ETH and SOL." action={<Badge tone="demo">API WIRING READY</Badge>}/>
    <section className="market-grid premium-market">
      {markets.map((m)=><article className="market-card premium-card" key={m.symbol}>
        <div className="market-top"><div><span className="symbol-label">{m.symbol}</span><small>PERPETUAL</small></div><Badge tone="neutral">{m.fresh}</Badge></div>
        <strong className="market-price">{m.price}</strong>
        <div className="mini-spark"><svg viewBox="0 0 180 48"><path d="M0 36 C25 30,35 38,55 27 S90 34,110 20 S145 28,180 10" fill="none" stroke="currentColor" strokeWidth="2"/></svg></div>
        <div className="market-details">
          <span>24h Change <b>{m.change}</b></span><span>Funding <b>{m.funding}</b></span><span>Mark / Index <b>{m.basis}</b></span><span>Regime <b>{m.regime}</b></span>
        </div>
      </article>)}
    </section>
    <section className="lower-grid">
      <Panel eyebrow="MARKET CONTEXT" title="Cross-asset comparison" action={<span className="muted-text">No synthetic values</span>}>
        <Table headers={['Symbol','Price','24h','Funding','Basis','Regime']} rows={markets.map(m=>[m.symbol,m.price,m.change,m.funding,m.basis,m.regime])}/>
      </Panel>
      <Panel eyebrow="REGIME" title="Context monitor"><PlaceholderChart/></Panel>
    </section>
  </>
}

function ModelPage(){
  return <>
    <Hero eyebrow="MODEL STATUS / SHADOW RUNNER" title="QuantSight V4" sub="Feature contract, thresholds, approval state and diagnostics." action={<Badge tone="shadow">SHADOW MODE</Badge>}/>
    <section className="stats-grid">
      <Stat label="Feature Version" value="v4_context_plus" detail="Frozen feature contract"/>
      <Stat label="Model Type" value="Directional classifier" detail="Probability-based decision"/>
      <Stat label="Approved Model" value="NOT READY" detail="Formal approval pending" tone="amber"/>
      <Stat label="Feature Freshness" value="Awaiting API" detail="Live age will appear here" tone="cyan"/>
    </section>
    <section className="main-grid">
      <Panel eyebrow="DECISION CONFIGURATION" title="Locked thresholds">
        <div className="threshold-grid">
          {[
            ['BTCUSDT','0.56','56%'],['ETHUSDT','0.56','56%'],['SOLUSDT','0.54','54%']
          ].map(([s,v,p])=><div className="threshold-card" key={s}><span>{s}</span><strong>{v}</strong><div className="threshold-bar"><i style={{width:p}}/></div><small>Directional probability gate</small></div>)}
        </div>
      </Panel>
      <Panel eyebrow="RUNTIME CONTRACT" title="Shadow status">
        <div className="detail-list">
          <div><span>Feature version</span><strong>v4_context_plus</strong></div>
          <div><span>Approval state</span><strong className="amber-text">NOT READY</strong></div>
          <div><span>Execution</span><strong>Orders OFF</strong></div>
          <div><span>Last signal update</span><strong>Awaiting API</strong></div>
        </div>
      </Panel>
    </section>
    <Panel eyebrow="MODEL DIAGNOSTICS" title="Evaluation workspace">
      <MetricStrip items={[
        {label:'Evaluated predictions',value:'—',hint:'Awaiting outcomes'},
        {label:'PASS directional precision',value:'—',hint:'No fake metrics'},
        {label:'REJECT accuracy',value:'—',hint:'No fake metrics'},
        {label:'Calibration / AUC',value:'—',hint:'Connect report source'},
      ]}/>
      <PlaceholderChart/>
    </Panel>
  </>
}

function SignalsPage({status}:{status:QuantStatus | null}){
  const rows = liveRows(status)

  const passCount =
    rows.filter(r=>r[3]==="PASS").length

  const staleCount =
    rows.filter(r=>r[3]==="STALE").length

  return <>
    <Hero
      eyebrow="SIGNAL MONITOR / DEMO FEED"
      title="Decision queue"
      sub="Directional model signals. Orders remain disabled."
      action={<Badge tone="off">ORDERS OFF</Badge>}
    />

    <section className="stats-grid">
      <Stat
        label="Monitored Symbols"
        value="3"
        detail="BTC · ETH · SOL"
        tone="green"
      />

      <Stat
        label="Shadow Freshness"
        value={status?.shadow?.fresh ? "FRESH" : "STALE"}
        detail="Backend status"
        tone={status?.shadow?.fresh ? "green" : "amber"}
      />

      <Stat
        label="PASS Signals"
        value={String(passCount)}
        detail="Current directional decisions"
        tone="green"
      />

      <Stat
        label="Stale Signals"
        value={String(staleCount)}
        detail="Freshness guard"
        tone="amber"
      />
    </section>

    <Panel
      eyebrow="LIVE SIGNALS"
      title="Signal ledger"
    >
      <Table
        headers={[
          "Symbol",
          "Probability Up",
          "Threshold",
          "Decision",
          "Feature Age",
          "Orders"
        ]}
        rows={rows}
      />
    </Panel>
  </>
}

function PaperPage(){
  return <>
    <Hero eyebrow="EXECUTION / DEMO ONLY" title="Paper trading control room" sub="Risk gates, dry-run state and demo execution readiness." action={<div className="environment"><Badge tone="demo">DEMO ONLY</Badge><span><LockKeyhole size={13}/> LIVE MONEY DISABLED</span></div>}/>
    <section className="stats-grid">
      <Stat label="Demo Auth" value="Connected" detail="Demo account context" tone="green"/>
      <Stat label="Paper Ready" value="Pending" detail="Full gate not completed" tone="amber"/>
      <Stat label="Orders Enabled" value="FALSE" detail="PLACE_ORDERS=False" tone="green"/>
      <Stat label="Open Positions" value="0" detail="Awaiting live demo state"/>
    </section>
    <section className="main-grid">
      <Panel eyebrow="RISK LIMITS" title="Execution guardrails">
        <div className="risk-grid">
          {[
            ['Max risk / trade','$15','Hard safety cap'],['Fixed demo risk','$10','Sizing target'],['Daily loss cap','$50','Closed-PnL guard'],['Max positions','1 / symbol','Concurrency guard']
          ].map(([a,b,c])=><div className="risk-tile" key={a}><span>{a}</span><strong>{b}</strong><small>{c}</small></div>)}
        </div>
      </Panel>
      <Panel eyebrow="SAFETY STATE" title="Order protection">
        <div className="detail-list">
          <div><span>Kill switch</span><strong className="green-text">OFF</strong></div>
          <div><span>Stale-feature guard</span><strong className="green-text">ACTIVE</strong></div>
          <div><span>Duplicate-signal guard</span><strong className="green-text">ACTIVE</strong></div>
          <div><span>Orders sent</span><strong>0</strong></div>
        </div>
      </Panel>
    </section>
    <Panel eyebrow="DEMO EXECUTION LEDGER" title="Recent activity">
      <Table headers={['Time','Symbol','Signal','Dry Run','Order','Result']} rows={[['—','—','Awaiting eligible PASS','—','OFF','No execution']]}/>
    </Panel>
  </>
}

function OutcomesPage(){
  return <>
    <Hero eyebrow="OUTCOME / JOURNAL" title="Prediction outcomes" sub="Directional evaluation ledger. Not profitability and not win rate."/>
    <section className="stats-grid outcomes-stats">
      <Stat label="Total Predictions" value="—" detail="Awaiting journal"/>
      <Stat label="Evaluated" value="—" detail="Awaiting outcomes"/>
      <Stat label="Pending" value="—" detail="Not evaluated" tone="amber"/>
      <Stat label="PASS Directional Precision" value="—" detail="Directional metric only"/>
      <Stat label="REJECT Accuracy" value="—" detail="Separate metric"/>
      <Stat label="Last Evaluated" value="—" detail="Timestamp pending"/>
    </section>
    <section className="lower-grid">
      <Panel eyebrow="EVALUATION LEDGER" title="Recent outcomes">
        <Table headers={['Symbol','Decision','Probability','Outcome','Future Return','State']} rows={[['—','—','—','—','—','Awaiting API']]}/>
      </Panel>
      <Panel eyebrow="EVALUATION TREND" title="Directional quality"><PlaceholderChart/></Panel>
    </section>
  </>
}

function RiskPage(){
  return <>
    <Hero eyebrow="RISK / SAFETY CONTROLS" title="Risk control center" sub="Hard limits and execution guards for demo trading." action={<Badge tone="green">SAFE MODE</Badge>}/>
    <section className="stats-grid">
      <Stat label="Daily Loss Cap" value="$50" detail="Closed-PnL guard" tone="green"/>
      <Stat label="Max Risk / Trade" value="$15" detail="Hard cap" tone="green"/>
      <Stat label="Max Positions / Symbol" value="1" detail="Concurrency limit" tone="green"/>
      <Stat label="Kill Switch" value="OFF" detail="Trading remains demo-only" tone="green"/>
    </section>
    <section className="main-grid">
      <Panel eyebrow="SAFETY CONTROLS" title="Guard status">
        <div className="guard-list">
          {[
            ['Stale Feature Guard','ACTIVE','Blocks aged signals'],
            ['Duplicate Signal Guard','ACTIVE','Prevents repeated execution'],
            ['Demo-only Endpoint','ACTIVE','Bybit Demo environment'],
            ['Order Placement','DISABLED','PLACE_ORDERS=False'],
          ].map(([a,b,c])=><div key={a}><span><ShieldCheck size={16}/></span><div><strong>{a}</strong><small>{c}</small></div><Badge tone="green">{b}</Badge></div>)}
        </div>
      </Panel>
      <Panel eyebrow="RISK EVENTS" title="Recent safety events">
        <div className="empty-state compact">No event stream connected yet.</div>
      </Panel>
    </section>
  </>
}

function SystemPage({status}:{status:QuantStatus | null}){
  const collector = Boolean(status?.live?.running)
  const builder = Boolean(status?.builder?.running)
  const shadow = Boolean(status?.shadow_proc?.running)
  const fresh = Boolean(status?.shadow?.fresh)

  return <>
    <Hero
      eyebrow="SYSTEM / RUNTIME HEALTH"
      title="System health"
      sub="Actual backend process and readiness state."
      action={
        <Badge
          tone={
            collector && builder && shadow && fresh
              ? "green"
              : "amber"
          }
        >
          {
            collector && builder && shadow && fresh
              ? "RUNTIME HEALTHY"
              : "RUNTIME DEGRADED"
          }
        </Badge>
      }
    />

    <section className="health-grid">

      <StatusCard
        icon={<DatabaseZap size={18}/>}
        label="Massive API"
        value={
          status?.massive_api_key
            ? "Connected"
            : "Missing"
        }
        detail="Backend environment"
        tone={
          status?.massive_api_key
            ? "green"
            : "amber"
        }
      />

      <StatusCard
        icon={<RefreshCcw size={18}/>}
        label="Collector"
        value={collector ? "Running" : "Stopped"}
        detail={
          collector
            ? `PID ${status?.live?.pid ?? "—"}`
            : "No active process"
        }
        tone={collector ? "green" : "amber"}
      />

      <StatusCard
        icon={<Boxes size={18}/>}
        label="Feature Builder"
        value={builder ? "Running" : "Stopped"}
        detail={
          builder
            ? `PID ${status?.builder?.pid ?? "—"}`
            : "No active process"
        }
        tone={builder ? "green" : "amber"}
      />

      <StatusCard
        icon={<Waypoints size={18}/>}
        label="Shadow Runner"
        value={shadow ? "Running" : "Stopped"}
        detail={
          shadow
            ? `PID ${status?.shadow_proc?.pid ?? "—"}`
            : "No active process"
        }
        tone={shadow ? "green" : "amber"}
      />

      <StatusCard
        icon={<TrendingUp size={18}/>}
        label="Funding Data"
        value={
          status?.readiness?.funding_data
            ? "Ready"
            : "Not Ready"
        }
        detail="Readiness milestone"
        tone={
          status?.readiness?.funding_data
            ? "green"
            : "amber"
        }
      />

      <StatusCard
        icon={<HardDrive size={18}/>}
        label="Historical Data"
        value={
          status?.readiness?.historical_data
            ? "Ready"
            : "Not Ready"
        }
        detail="Readiness milestone"
        tone={
          status?.readiness?.historical_data
            ? "green"
            : "amber"
        }
      />

    </section>

    <section className="main-grid">

      <Panel eyebrow="RUNTIME" title="Process state">
        <div className="detail-list">
          <div>
            <span>Collector</span>
            <strong>{collector ? "RUNNING" : "STOPPED"}</strong>
          </div>

          <div>
            <span>Feature Builder</span>
            <strong>{builder ? "RUNNING" : "STOPPED"}</strong>
          </div>

          <div>
            <span>Shadow Process</span>
            <strong>{shadow ? "RUNNING" : "STOPPED"}</strong>
          </div>

          <div>
            <span>Shadow Freshness</span>
            <strong>{fresh ? "FRESH" : "STALE"}</strong>
          </div>
        </div>
      </Panel>

      <Panel eyebrow="MODEL" title="Runtime contract">
        <div className="detail-list">
          <div>
            <span>Feature Version</span>
            <strong>
              {status?.features?.current_version ?? "—"}
            </strong>
          </div>

          <div>
            <span>Model</span>
            <strong>
              {status?.shadow?.status?.model ?? "—"}
            </strong>
          </div>

          <div>
            <span>Validation</span>
            <strong>
              {status?.validation?.overall ?? "—"}
            </strong>
          </div>

          <div>
            <span>Orders</span>
            <strong>OFF</strong>
          </div>
        </div>
      </Panel>

    </section>
  </>
}

function EvidencePage(){
  return <>
    <Hero eyebrow="SYSTEM / EVIDENCE" title="Data & evidence" sub="Model artifacts, journals, reports and audit-friendly runtime evidence." action={<Badge tone="demo">READ ONLY</Badge>}/>
    <section className="evidence-grid">
      {[
        [BookOpenCheck,'Prediction Journal','Records','Awaiting API','Append-only signal history'],
        [ListChecks,'Outcome Evaluator','Evaluated','Awaiting API','30m directional outcomes'],
        [FileText,'Model Reports','Reports','Available locally','Training and diagnostics'],
        [Binary,'Feature Manifest','Version','v4_context_plus','Frozen feature contract'],
        [ShieldAlert,'Validation Evidence','State','CHECK','Readiness artifacts'],
        [ServerCog,'Runtime Logs','Source','Local runtime','Operational evidence'],
      ].map(([Icon,title,k,v,d]:any)=><article className="evidence-card" key={title}>
        <div className="evidence-icon"><Icon size={18}/></div>
        <div><span>{title}</span><strong>{v}</strong><small>{d}</small></div>
        <div className="evidence-meta"><span>{k}</span><b>→</b></div>
      </article>)}
    </section>
    <section className="lower-grid">
      <Panel eyebrow="ARTIFACT INDEX" title="Available evidence">
        <Table headers={['Artifact','State','Source']} rows={[
          ['Journal','API pending','ml/v5_journal'],
          ['Outcomes','API pending','ml/v5_journal'],
          ['Feature manifest','Local','ml/features/v4_context_plus'],
          ['Training reports','Local','ml/reports'],
        ]}/>
      </Panel>
      <Panel eyebrow="AUDIT NOTES" title="Evidence policy">
        <div className="note-box">Directional metrics are kept separate from profitability. Model approval remains <b>NOT READY</b> until formal promotion.</div>
      </Panel>
    </section>
  </>
}

function SettingsPage(){
  return <>
    <Hero eyebrow="SYSTEM / DISPLAY PREFERENCES" title="Settings" sub="Frontend preferences only. No secrets or API keys are exposed."/>
    <section className="settings-grid">
      <Panel eyebrow="APPEARANCE" title="Display">
        <div className="settings-list">
          <label>Theme<select defaultValue="Dark"><option>Dark</option></select></label>
          <label>Table density<select defaultValue="Compact"><option>Compact</option><option>Comfortable</option></select></label>
          <label>Timestamp format<select defaultValue="UTC"><option>UTC</option><option>Local</option></select></label>
        </div>
      </Panel>
      <Panel eyebrow="REFRESH" title="Runtime display">
        <div className="settings-list">
          <label>Refresh interval<select defaultValue="30 seconds"><option>15 seconds</option><option>30 seconds</option><option>60 seconds</option></select></label>
          <label>Default page<select defaultValue="Dashboard"><option>Dashboard</option><option>Signals</option><option>System Health</option></select></label>
        </div>
      </Panel>
    </section>
  </>
}

function Dashboard({status}:{status:QuantStatus | null}){
  const rows = liveRows(status)

  const collector = Boolean(status?.live?.running)
  const builder = Boolean(status?.builder?.running)
  const shadow = Boolean(status?.shadow_proc?.running)
  const fresh = Boolean(status?.shadow?.fresh)

  const healthy =
    collector &&
    builder &&
    shadow &&
    fresh

  return <>
    <Hero
      eyebrow="DEMO MARKET INTELLIGENCE"
      title="QuantSight Market Intelligence"
      sub="Live model signals, market context, and demo trading readiness."
      action={
        <div className="environment">
          <Badge tone="demo">
            DEMO PAPER TRADING
          </Badge>

          <span>
            <LockKeyhole size={13}/>
            LIVE MONEY DISABLED
          </span>
        </div>
      }
    />

    <section className="stats-grid">

      <Stat
        label="Model / Feature Version"
        value={
          status?.features?.current_version ??
          "v4_context_plus"
        }
        detail="Feature contract active"
      />

      <Stat
        label="Live Signal State"
        value="3 Symbols Monitored"
        detail={
          fresh
            ? "Shadow feed fresh"
            : "Shadow feed stale"
        }
        tone={fresh ? "green" : "amber"}
      />

      <Stat
        label="Demo Exposure"
        value="$0.00"
        detail="Orders disabled"
        tone="amber"
      />

      <Stat
        label="Runtime"
        value={healthy ? "HEALTHY" : "DEGRADED"}
        detail={
          `C:${collector?"ON":"OFF"} · ` +
          `B:${builder?"ON":"OFF"} · ` +
          `S:${shadow?"ON":"OFF"}`
        }
        tone={healthy ? "green" : "amber"}
      />

    </section>

    <section className="main-grid">

      <Panel
        eyebrow="LIVE SIGNALS / DEMO FEED"
        title="Decision queue"
        action={
          <span className="streaming">
            <Activity size={13}/>
            Live API
          </span>
        }
      >

        <Table
          headers={[
            "Symbol",
            "Probability Up",
            "Threshold",
            "Decision",
            "Feature Age",
            "Orders"
          ]}
          rows={rows}
        />

      </Panel>

      <Panel
        eyebrow="MODEL STATUS"
        title="QuantSight V4"
        action={
          <Badge tone="shadow">
            SHADOW MODE
          </Badge>
        }
      >

        <div className="approval">
          <span className="approval-icon">✓</span>

          <div>
            <span>Approved Model</span>
            <strong>
              {
                Object.keys(
                  status?.approved ?? {}
                ).length
                  ? "APPROVED"
                  : "NOT READY"
              }
            </strong>
          </div>
        </div>

        <div className="detail-list">

          <div>
            <span>Feature Version</span>
            <strong>
              {status?.features?.current_version ?? "—"}
            </strong>
          </div>

          <div>
            <span>Model Type</span>
            <strong>
              {status?.shadow?.status?.model ?? "—"}
            </strong>
          </div>

          <div>
            <span>Shadow Freshness</span>
            <strong>
              {fresh ? "FRESH" : "STALE"}
            </strong>
          </div>

        </div>

      </Panel>

    </section>

    <Panel
      eyebrow="READINESS"
      title="Runtime milestones"
      action={
        <span className="muted-text">
          {status?.progress?.percent ?? "—"}%
        </span>
      }
    >

      <div className="market-grid">

        <div className="market-card">
          <div className="market-top">
            <strong>Historical Data</strong>
            <span>
              {
                status?.readiness?.historical_data
                  ? "PASS"
                  : "NOT READY"
              }
            </span>
          </div>

          <strong className="market-price">
            {
              status?.rows?.BTCUSDT?.["5m"]?.rows ??
              "—"
            }
          </strong>

          <div className="market-details">
            <span>BTC 5m rows</span>
          </div>
        </div>

        <div className="market-card">
          <div className="market-top">
            <strong>Funding Data</strong>
            <span>
              {
                status?.readiness?.funding_data
                  ? "PASS"
                  : "NOT READY"
              }
            </span>
          </div>

          <strong className="market-price">
            {
              status?.rows?.BTCUSDT?.funding?.rows ??
              "—"
            }
          </strong>

          <div className="market-details">
            <span>BTC funding rows</span>
          </div>
        </div>

        <div className="market-card">
          <div className="market-top">
            <strong>Shadow Freshness</strong>
            <span>{fresh ? "FRESH" : "STALE"}</span>
          </div>

          <strong className="market-price">
            {fresh ? "YES" : "NO"}
          </strong>

          <div className="market-details">
            <span>
              Orders <b>OFF</b>
            </span>
          </div>
        </div>

      </div>

    </Panel>
  </>
}

function PageContent({route,status}:{route:string;status:QuantStatus | null}){
  if(route==='/market') return <MarketPage/>
  if(route==='/ai-model') return <ModelPage/>
  if(route==='/signals') return <SignalsPage status={status}/>
  if(route==='/paper-trading') return <PaperPage/>
  if(route==='/outcomes') return <OutcomesPage/>
  if(route==='/risk') return <RiskPage/>
  if(route==='/system-health') return <SystemPage status={status}/>
  if(route==='/data-evidence') return <EvidencePage/>
  if(route==='/settings') return <SettingsPage/>
  return <Dashboard status={status}/>
}

export default function Page(){
  const pathname=usePathname()
  const route=pathname==='/'?'/dashboard':pathname
  const [mobileOpen,setMobileOpen]=useState(false)

  const [status,setStatus] =
    useState<QuantStatus | null>(null)

  useEffect(()=>{
    let alive = true

    async function load(){
      try{
        const res =
          await fetch("/api/status",{
            cache:"no-store"
          })

        const payload =
          await res.json()

        if(alive){
          setStatus(
            payload?.ok
              ? payload.data
              : null
          )
        }
      }catch{
        if(alive){
          setStatus(null)
        }
      }
    }

    load()

    const id =
      setInterval(load,15000)

    return ()=>{
      alive = false
      clearInterval(id)
    }
  },[])
  const active=navItems.find(([,href])=>href===route)?.[0]??'Dashboard'

  return <div className="terminal-shell">
    <aside className={`sidebar ${mobileOpen?'sidebar-open':''}`}>
      <div className="brand">
        <div className="brand-mark"><Sparkles size={16}/></div>
        <div><strong>QuantSight</strong><span>AI MARKET INTELLIGENCE</span></div>
        <button className="close-menu" onClick={()=>setMobileOpen(false)} aria-label="Close menu"><X size={18}/></button>
      </div>
      <div className="workspace"><span className="status-dot"/> DEMO ENVIRONMENT <ChevronDown size={14}/></div>
      <nav aria-label="Primary navigation">
        <p className="nav-label">Workspace</p>
        {navItems.slice(0,3).map(([label,href,Icon])=><Link key={href} href={href} onClick={()=>setMobileOpen(false)} className={`nav-item ${active===label?'active':''}`}><Icon size={17}/><span>{label}</span></Link>)}
        <p className="nav-label nav-label-spaced">Operations</p>
        {navItems.slice(3,8).map(([label,href,Icon])=><Link key={href} href={href} onClick={()=>setMobileOpen(false)} className={`nav-item ${active===label?'active':''}`}><Icon size={17}/><span>{label}</span>{label==='Signals'&&<span className="nav-badge">3</span>}</Link>)}
        <p className="nav-label nav-label-spaced">System</p>
        {navItems.slice(8).map(([label,href,Icon])=><Link key={href} href={href} onClick={()=>setMobileOpen(false)} className={`nav-item ${active===label?'active':''}`}><Icon size={17}/><span>{label}</span></Link>)}
      </nav>
      <div className="sidebar-footer"><div className="avatar">AC</div><div><strong>Analyst Console</strong><span>Demo access</span></div><SlidersHorizontal size={16}/></div>
    </aside>

    <main className="main-content">
      <header className="topbar">
        <button className="menu-button" onClick={()=>setMobileOpen(true)} aria-label="Open menu"><Menu size={20}/></button>
        <div><p className="eyebrow">MARKET INTELLIGENCE / {String(active).toUpperCase()}</p><h1>{active}</h1></div>
        <div className="top-actions">
          <div className="health">
            <span className="status-dot"/>
            System Health
            <b>
              {
                status?.live?.running &&
                status?.builder?.running &&
                status?.shadow_proc?.running &&
                status?.shadow?.fresh
                  ? "Operational"
                  : "Degraded"
              }
            </b>
          </div>
          <div className="health">
            <Clock3 size={14}/>
            Data Freshness
            <b>{status?.shadow?.fresh ? "Fresh" : "Stale"}</b>
          </div>
          <Badge tone="off">ORDERS: OFF</Badge>
          <button className="help" aria-label="Help"><CircleHelp size={19}/></button>
        </div>
      </header>
      <div className="content-wrap"><PageContent route={route} status={status}/></div>
    </main>
  </div>
}
