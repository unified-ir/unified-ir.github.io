"use strict";(self.webpackChunkunified_ir=self.webpackChunkunified_ir||[]).push([[342],{1103:(e,n,i)=>{i.r(n),i.d(n,{assets:()=>o,contentTitle:()=>c,default:()=>p,frontMatter:()=>d,metadata:()=>l,toc:()=>a});var r=i(4848),s=i(8453);const d={sidebar_position:1},c="Introduction To Unified IR",l={id:"tutorial-basics/unified-ir-defs",title:"Introduction To Unified IR",description:"Overview",source:"@site/docs/tutorial-basics/1-unified-ir-defs.md",sourceDirName:"tutorial-basics",slug:"/tutorial-basics/unified-ir-defs",permalink:"/tutorial-basics/unified-ir-defs",draft:!1,unlisted:!1,tags:[],version:"current",sidebarPosition:1,frontMatter:{sidebar_position:1},sidebar:"tutorialSidebar",previous:{title:"Unified IR - Basics",permalink:"/category/unified-ir---basics"}},o={},a=[{value:"Overview",id:"overview",level:2},{value:"Kernel Graph",id:"kernel-graph",level:2},{value:"iGraph",id:"igraph",level:2},{value:"Hardware Model",id:"hardware-model",level:3},{value:"Parallel Description",id:"parallel-description",level:3},{value:"iSlice",id:"islice",level:3},{value:"iOp",id:"iop",level:3},{value:"\u8bbf\u5b58iOp",id:"\u8bbf\u5b58iop",level:4},{value:"\u8ba1\u7b97iOp",id:"\u8ba1\u7b97iop",level:4}];function t(e){const n={code:"code",h1:"h1",h2:"h2",h3:"h3",h4:"h4",header:"header",p:"p",pre:"pre",strong:"strong",...(0,s.R)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(n.header,{children:(0,r.jsx)(n.h1,{id:"introduction-to-unified-ir",children:"Introduction To Unified IR"})}),"\n",(0,r.jsx)(n.h2,{id:"overview",children:"Overview"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"Unified IR"})," \u662f\u4e00\u4e2a\u63cf\u8ff0\u5f20\u91cf\u7a0b\u5e8f\u7684\u7edf\u4e00\u4e2d\u95f4\u8868\u793a\uff0c\u65e8\u5728\u4e3a\u4eba\u5de5\u667a\u80fd\u5e94\u7528\u63d0\u4f9b\u8de8\u5e73\u53f0\u7684\u7edf\u4e00\u7f16\u8bd1\u4f18\u5316\u3002\u6bcf\u4e2a\u5f20\u91cf\u7a0b\u5e8f\u90fd\u5bf9\u5e94\u4e8e ",(0,r.jsx)(n.strong,{children:"Unified IR"})," \u4e2d\u7684\u4e00\u4e2a ",(0,r.jsx)(n.strong,{children:"Kernel Graph"}),"\uff0c\u63cf\u8ff0\u4e86\u4e0d\u540c\u7684\u8bbe\u5907\u7b97\u5b50\u5728\u5f20\u91cf\u7a0b\u5e8f\u4e2d\u7684\u6570\u636e\u4f9d\u8d56\u5173\u7cfb\uff1b",(0,r.jsx)(n.strong,{children:"Kernel Graph"})," \u4e2d\u90e8\u5206\u8282\u70b9\u4e3a\u81ea\u5b9a\u4e49\u7b97\u5b50\uff0c\u8868\u793a\u4e3a\u4e00\u4e2a ",(0,r.jsx)(n.strong,{children:"iGraph"}),"\uff0c\u8fd9\u662f\u4e00\u4e2a\u5bf9\u63cf\u8ff0\u7b97\u5b50\u5b9e\u73b0\u7684\u6307\u4ee4\u7ea7\u4e2d\u95f4\u8868\u793a\u3002"]}),"\n",(0,r.jsx)(n.h2,{id:"kernel-graph",children:"Kernel Graph"}),"\n",(0,r.jsxs)(n.p,{children:["\u5728 ",(0,r.jsx)(n.strong,{children:"Unified IR"})," \u4e2d\uff0c",(0,r.jsx)(n.strong,{children:"Kernel Graph"})," \u7528\u4e8e\u63cf\u8ff0\u4e00\u4e2a\u5f20\u91cf\u7a0b\u5e8f\uff0c\u5176\u4e2d\u7684\u6bcf\u4e2a\u8282\u70b9\u8868\u793a\u4e00\u4e2a\u5728\u76ee\u6807\u786c\u4ef6\u5e73\u53f0\u4e0a\u6267\u884c\u7684\u7b97\u5b50\uff0c\u6bcf\u6761\u8fb9\u8868\u793a\u5728\u4e0d\u540c\u7b97\u5b50\u4e4b\u95f4\u5171\u4eab\u7684\u5f20\u91cf\u3002\u6240\u6709\u7684\u5f20\u91cf\u90fd\u5b58\u50a8\u5728\u8bbe\u5907\u7684DRAM\u4e2d\uff0c\u8fd9\u662f\u56e0\u4e3a\u4e0d\u540c\u7684\u7b97\u5b50\u65e0\u6cd5\u901a\u8fc7\u5bc4\u5b58\u5668\u6587\u4ef6\u6216\u8005SRAM\u5171\u4eab\u6570\u636e\u3002\u6bcf\u4e2a\u8282\u70b9\u6240\u8868\u793a\u7684\u7b97\u5b50\u53ef\u4ee5\u662f\u5bf9\u5e94\u786c\u4ef6\u9884\u5b9a\u4e49\u7684\u7b97\u5b50\uff0c\u5982\u77e9\u9635\u4e58\u6cd5\u548c\u5377\u79ef\uff0c\u4e5f\u53ef\u4ee5\u662f\u7531 ",(0,r.jsx)(n.strong,{children:"iGraph"})," \u63cf\u8ff0\u7684\u81ea\u5b9a\u4e49\u7b97\u5b50\uff0c",(0,r.jsx)(n.strong,{children:"iGraph"})," \u5141\u8bb8\u6211\u4eec\u5728 ",(0,r.jsx)(n.strong,{children:"Kernel Graph"})," \u5c42\u7ea7\u4e0a\u505a\u7b97\u5b50\u878d\u5408\u7b49\u7b97\u5b50\u95f4\uff08inter-operator\uff09\u4f18\u5316\u3002"]}),"\n",(0,r.jsx)(n.h2,{id:"igraph",children:"iGraph"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"iGraph"})," \u4ece\u786c\u4ef6\u6307\u4ee4\u7684\u89d2\u5ea6\u63cf\u8ff0\u4e86\u7b97\u5b50\u7684\u5b9e\u73b0\uff0c\u5b83\u7684\u5f62\u5f0f\u662f\u4e00\u4e2a\u6570\u636e\u6d41\u56fe\uff0c\u63cf\u8ff0\u4e86\u6570\u636e\u5728\u4e0d\u540c\u5185\u5b58\u5c42\u7ea7\u95f4\u7684\u79fb\u52a8\u548c\u786c\u4ef6\u6307\u4ee4\u5bf9\u6570\u636e\u7684\u5904\u7406\u3002\u56fe\u4e2d\u7684\u6bcf\u4e2a\u8282\u70b9\u4e3a ",(0,r.jsx)(n.strong,{children:"iOp"}),"\uff0c\u8868\u793a\u5728\u786c\u4ef6\u7684\u6700\u5c0f\u5e76\u884c\u5355\u5143\u4e0a\u7684\u4e00\u6761\u786c\u4ef6\u6307\u4ee4\u6216\u662f\u4e00\u4e2a\u6307\u4ee4\u5e8f\u5217\uff0c\u6bcf\u6761\u8fb9\u662f\u4e00\u4e2a ",(0,r.jsx)(n.strong,{children:"iSlice"}),"\uff0c\u8868\u793a\u6570\u636e\u5207\u7247\uff0c\u662f\u4e0a\u8ff0\u7684 ",(0,r.jsx)(n.strong,{children:"iOp"})," \u6240\u4f5c\u7528\u7684\u57fa\u672c\u5355\u5143\u3002",(0,r.jsx)(n.strong,{children:"iGraph"})," \u8fd8\u5305\u542b\u786c\u4ef6\u5e76\u884c\u67b6\u6784\u4fe1\u606f\u548c\u5bf9\u7b97\u5b50\u5b9e\u73b0\u5e76\u884c\u65b9\u5f0f\u7684\u63cf\u8ff0\u3002"]}),"\n",(0,r.jsx)(n.h3,{id:"hardware-model",children:"Hardware Model"}),"\n",(0,r.jsxs)(n.p,{children:["\u8003\u8651\u5230AI\u52a0\u901f\u786c\u4ef6\u5185\u5728\u7684\u76f8\u4f3c\u6027\uff0c",(0,r.jsx)(n.strong,{children:"Unified IR"})," \u63d0\u51fa\u4e86\u4e00\u5957\u9ad8\u5c42\u6b21\u7684\u5bf9\u786c\u4ef6\u5e76\u884c\u67b6\u6784\u7684\u5efa\u6a21\uff0c\u4ece\u800c\u53ef\u4ee5\u7528\u4e00\u5957\u7edf\u4e00\u7684\u65b9\u5f0f\u63cf\u8ff0\u4e0d\u540c\u7684\u786c\u4ef6\u7684\u7279\u6027\u3002"]}),"\n",(0,r.jsxs)(n.p,{children:["\u6211\u4eec\u5b9a\u4e49 ",(0,r.jsx)(n.strong,{children:"\u6700\u5c0f\u5e76\u884c\u5355\u5143"})," \u4e3a\u76ee\u6807\u786c\u4ef6\u4e2d\u53ef\u4ee5\u72ec\u7acb\u6267\u884c\u7684\u6700\u5c0f\u5355\u5143\uff0c\u8fd9\u5728\u4e0d\u540c\u7684\u786c\u4ef6\u4e2d\u6709\u4e0d\u540c\u7684\u542b\u4e49\uff0c\u5728NVIDIA\u7cfb\u5217\u7684GPU\u4e0a\u5bf9\u5e94\u4e8eSMSP\uff08\u53ef\u4ee5\u72ec\u7acb\u7684\u8c03\u5ea6warp\uff09\uff0c\u800c\u5728Ascend 910\u7cfb\u5217\u82af\u7247\u4e2d\u5219\u5bf9\u5e94\u4e8e\u4e00\u4e2aAI Core\u3002\u5728\u90e8\u5206\u786c\u4ef6\u4e2d\uff0c\u591a\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u53ef\u4ee5\u7ec4\u6210\u4e00\u7ec4\u534f\u540c\u8ba1\u7b97\uff0c\u4e4b\u95f4\u53ef\u4ee5\u901a\u8fc7SRAM\u6765\u8fdb\u884c\u901a\u4fe1\uff0c\u6211\u4eec\u79f0\u8fd9\u6837\u7684\u4e00\u7ec4\u6700\u5c0f\u5e76\u884c\u5355\u5143\u4e3a ",(0,r.jsx)(n.strong,{children:"\u5e76\u884c\u7ec4"}),"\u3002\u4f8b\u5982\u5728CUDA\u7f16\u7a0b\u4e2d\uff0c\u4e00\u4e2a\u7ebf\u7a0b\u5757\u5305\u542b\u591a\u4e2aWarp\uff0c\u5b83\u4eec\u90fd\u5728\u540c\u4e00\u4e2aSM\u4e0a\u6267\u884c\uff0c\u5f7c\u6b64\u4e4b\u95f4\u53ef\u4ee5\u8fdb\u884c\u540c\u6b65\u548c\u901a\u8fc7\u5171\u4eab\u5185\u5b58\u540c\u6b65\u6570\u636e\u3002\u800c\u4e00\u4e2aAI\u52a0\u901f\u786c\u4ef6\u5305\u542b\u591a\u4e2a\u4e0a\u8ff0\u7684 ",(0,r.jsx)(n.strong,{children:"\u5e76\u884c\u7ec4"}),"\u3002"]}),"\n",(0,r.jsxs)(n.p,{children:["\u5bf9\u5185\u5b58\u5c42\u7ea7\u4e5f\u53ef\u4ee5\u505a\u7c7b\u4f3c\u7684\u5212\u5206\uff0c\u6309\u7167\u8bbf\u95ee\u901f\u5ea6\u4ece\u9ad8\u5230\u4f4e\u53ef\u4ee5\u5206\u4e3a ",(0,r.jsx)(n.strong,{children:"REG"}),"\u3001",(0,r.jsx)(n.strong,{children:"SRAM"})," \u548c ",(0,r.jsx)(n.strong,{children:"DRAM"}),"\u3002",(0,r.jsx)(n.strong,{children:"REG"})," \u5c42\u7ea7\u7684\u5185\u5b58\u7531\u6bcf\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u72ec\u5360\uff0c\u8bbf\u95ee\u901f\u5ea6\u6700\u5feb\uff1b",(0,r.jsx)(n.strong,{children:"SRAM"})," \u5c42\u7ea7\u7684\u5185\u5b58\u5219\u53ef\u4ee5\u5728\u4e0d\u540c\u7684\u5e76\u884c\u7ec4\u4e4b\u95f4\u5171\u4eab\uff0c\u8bbf\u95ee\u901f\u5ea6\u6b21\u4e4b\uff1b",(0,r.jsx)(n.strong,{children:"DRAM"})," \u5728\u4e0d\u540c\u7684\u7b97\u5b50\u4e4b\u95f4\u5171\u4eab\uff0c\u8bbf\u95ee\u901f\u5ea6\u6700\u6162\u3002"]}),"\n",(0,r.jsxs)(n.p,{children:["\u56e0\u6b64\uff0c\u6839\u636e\u4e0a\u8ff0\u7684 \u201c\u786c\u4ef6 -> \u5e76\u884c\u7ec4 -> \u6700\u5c0f\u5e76\u884c\u5355\u5143\u201d \u7684\u62bd\u8c61\uff0c\u6211\u4eec\u53ef\u4ee5\u7528\u4e8c\u5143\u7ec4 (numGroup, numUnit) \u6765\u63cf\u8ff0\u786c\u4ef6\u7684\u5e76\u884c\u67b6\u6784\uff0c\u5206\u522b\u8868\u793a\u4e00\u4e2a\u786c\u4ef6\u4e2d\u6709\u591a\u5c11\u4e2a\u5e76\u884c\u7ec4\uff0c\u4e00\u4e2a\u5e76\u884c\u7ec4\u4e2d\u6709\u591a\u5c11\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u3002\u7531\u4e8eCUDA\u7684\u7f16\u7a0b\u63a5\u53e3\u5bf9\u5e95\u5c42\u7684\u786c\u4ef6\u67b6\u6784\u505a\u4e86\u8fdb\u4e00\u6b65\u7684\u62bd\u8c61\uff0c\u4e00\u4e2a\u6838\u51fd\u6570\u7684grid size\u548cblock size\u5747\u53ef\u53d8\uff0c\u6211\u4eec\u8fd9\u5957\u6a21\u578b\u5728\u63cf\u8ff0CUDA\u7b97\u5b50\u65f6\u9700\u8981\u505a\u4e00\u4e9b\u5fae\u5999\u7684\u8c03\u6574\uff1a\u6211\u4eec\u4ee4numGroup\u5bf9\u5e94\u4e8e\u7b97\u5b50\u5b9e\u73b0\u4e2d\u7684grid size\uff0c\u800cnumUnit\u5bf9\u5e94\u4e8e\u7ebf\u7a0b\u5757\u4e2d\u7684Warp\u6570\u91cf\u3002\u6b64\u5916\uff0cAscend 910 \u7cfb\u5217\u82af\u7247\u7684AI Core\u4e4b\u95f4\u4e0d\u80fd\u8fdb\u884c\u901a\u4fe1\uff0c\u56e0\u6b64\u5c31\u4e0d\u5b58\u5728\u4e0a\u8ff0\u7684 ",(0,r.jsx)(n.strong,{children:"\u5e76\u884c\u7ec4"})," \u5c42\u7ea7\uff0c\u8fd9\u5bf9\u5e94\u4e8enumUnit\u4e3a1\u7684\u9000\u5316\u60c5\u51b5\u3002"]}),"\n",(0,r.jsxs)(n.p,{children:["\u7c7b\u4f3c\u7684\uff0c\u6211\u4eec\u4e5f\u53ef\u4ee5\u5c06\u5177\u4f53\u786c\u4ef6\u7684\u5185\u5b58\u5c42\u7ea7\u6309\u7167\u5176\u548c\u5e76\u884c\u67b6\u6784\u7684\u5173\u7cfb\u6620\u5c04\u5230\u4e0a\u8ff0\u7684 \u201cREG -> SRAM -> DRAM\u201d \u5c42\u7ea7\u4e0a\u3002\u5bf9\u4e8eNVIDIA\u7684GPU\u6765\u8bf4\uff0c\u5bc4\u5b58\u5668\u6587\u4ef6\u5bf9\u5e94 ",(0,r.jsx)(n.strong,{children:"REG"}),"\uff0c\u5171\u4eab\u5185\u5b58\u5bf9\u5e94 ",(0,r.jsx)(n.strong,{children:"SRAM"}),"\uff0c\u5168\u5c40\u5185\u5b58\u5bf9\u5e94 ",(0,r.jsx)(n.strong,{children:"DRAM"}),"\u3002"]}),"\n",(0,r.jsx)(n.h3,{id:"parallel-description",children:"Parallel Description"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"iGraph"})," \u4e2d\u7684\u5e76\u884c\u65b9\u5f0f\u63cf\u8ff0\u5305\u542b\u4e24\u4e2a\u7ef4\u5ea6\uff0c\u5206\u522b\u662f\u5e76\u884c\u7ef4\u5ea6\u548c\u5faa\u73af\u7ef4\u5ea6\uff0c\u7528\u5143\u7ec4 (numParallel, numLoop) \u8868\u793a\u3002\u524d\u8005\u8868\u793a\u7b97\u5b50\u5b9e\u73b0\u4e2d\u5c06\u8ba1\u7b97\u4efb\u52a1\u5728 numParallel \u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u4e2d\u5e76\u884c\u6267\u884c\uff0c\u5e76\u4e3a\u6bcf\u4e00\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u5206\u914d\u4e00\u4e2a\u53d6\u503c\u4e8e [0, numParallel) \u7684\u548c\u786c\u4ef6\u5e76\u884c\u67b6\u6784\u76f8\u5bb9\u7684\u5e76\u884c\u7f16\u53f7\u3002\u4f8b\u5982\uff0c\u5bf9\u4e8e\u786c\u4ef6\u67b6\u6784\u7531 (108, 4) \u6765\u63cf\u8ff0\u7684\u786c\u4ef6\uff0c\u5176 numParallel = $108 \\times 4$ = $432$\uff0c\u7b2c1\u4e2a\u5e76\u884c\u7ec4\u4e2d\u76844\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u7684\u7f16\u53f7\u4e3a0\u30011\u30012\u30013\uff0c\u7b2c2\u4e2a\u5e76\u884c\u7ec4\u4e3a4\u30015\u30016\u30017\uff0c\u4ee5\u6b64\u7c7b\u63a8\u3002\u7f16\u53f7\u7528\u4e8e\u4e3a\u4e0d\u540c\u7684\u5355\u5143\u5206\u914d\u8ba1\u7b97\u4efb\u52a1\uff0c\u5176\u76f8\u5bb9\u6027\u8ba9\u6211\u4eec\u53ef\u4ee5\u5728\u5e76\u884c\u7ec4\u7684\u5c42\u7ea7\u4e0a\u5bf9\u6570\u636e\u7684\u590d\u7528\u8fdb\u884c\u4f18\u5316\u3002\u540e\u8005\u8868\u793a\u4e86\u6bcf\u4e2a\u5e76\u884c\u5904\u7406\u5355\u5143\u5185\u4f1a\u505a\u957f\u5ea6\u4e3anumLoop\u7684\u4e32\u884c\u5faa\u73af\uff0c\u6267\u884c\u7531 ",(0,r.jsx)(n.strong,{children:"iGraph"})," \u5185\u7684 ",(0,r.jsx)(n.strong,{children:"iOp"})," \u6240\u5b9a\u4e49\u7684\u6307\u4ee4\u5e8f\u5217\u3002\u5176\u8bed\u4e49\u5982\u4e0b\uff1a"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-python",children:"for parallel_id in 0...numParallel: # Parallel\n    for loop_id in 0...numLoop: # Series\n        loopBody(parallel_id, loop_id)\n"})}),"\n",(0,r.jsx)(n.h3,{id:"islice",children:"iSlice"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"iSlice"})," \u7528\u4e8e\u8868\u793a\u6bcf\u4e2a\u5e76\u884c\u5904\u7406\u5355\u5143\u5728\u6bcf\u4e2a\u5faa\u73af\u4e2d\u6240\u9700\u8981\u5904\u7406\u7684\u6570\u636e\u5207\u7247\uff0c\u63cf\u8ff0\u4e86\u6570\u636e\u5982\u4f55\u5728\u4e0d\u540c\u7684\u5e76\u884c\u5355\u5143\u548c\u4e0d\u540c\u7684\u5faa\u73af\u4e0a\u8fdb\u884c\u5212\u5206\u3002\u6240\u6709 ",(0,r.jsx)(n.strong,{children:"iSlice"})," \u90fd\u57fa\u4e8e\u4e00\u4e2a ",(0,r.jsx)(n.strong,{children:"iPointer"}),"\uff0c",(0,r.jsx)(n.strong,{children:"iPointer"})," \u8868\u793a\u5728\u67d0\u5185\u5b58\u5c42\u7ea7\u4e0a\u7684\u4e00\u5757\u8fde\u7eed\u5185\u5b58\uff0c\u5305\u542b\u6240\u5728\u5c42\u7ea7\u3001\u540d\u79f0\u3001\u6570\u636e\u7c7b\u578b\u548c\u957f\u5ea6\u7b49\u5c5e\u6027\u3002\u5176\u4e2d\uff0c",(0,r.jsx)(n.code,{children:"dram"})," \u4e0a\u7684iPointer\u5728\u5168\u5c40\u552f\u4e00\uff0c",(0,r.jsx)(n.code,{children:"sram"})," \u4e0a\u7684iPointer\u5728\u6bcf\u4e2a\u5e76\u884c\u7ec4\u5185\u552f\u4e00\uff0c\u800c ",(0,r.jsx)(n.code,{children:"reg"})," \u4e0a\u7684iPointer\u5219\u4e3a\u6bcf\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u72ec\u5360\u3002",(0,r.jsx)(n.strong,{children:"iSlice"})," \u5219\u8868\u793a\u5728\u5b83\u6240\u57fa\u4e8e\u7684 ",(0,r.jsx)(n.strong,{children:"iPointer"})," \u6240\u6307\u5411\u7684\u8fde\u7eed\u5185\u5b58\u4e0a\u7684\u4e00\u4e2a\u4e8c\u7ef4\u5207\u7247\uff0c\u7531\u4e00\u7ec4\u8fde\u7eed\u7684\u5185\u5b58\u7247\u6bb5\u7ec4\u6210\uff0c\u5177\u6709\u5c5e\u6027 ",(0,r.jsx)(n.code,{children:"Shape"}),"\u3001",(0,r.jsx)(n.code,{children:"Stride"})," \u548c ",(0,r.jsx)(n.code,{children:"Offset"}),"\u3002",(0,r.jsx)(n.code,{children:"Shape"})," \u662f\u4e00\u4e2a\u4e8c\u5143\u7ec4\uff0c\u8868\u793a\u8be5\u5185\u5b58\u5207\u7247\u7684\u5f62\u72b6\uff0c",(0,r.jsx)(n.code,{children:"Stride"})," \u4e5f\u662f\u4e00\u4e2a\u4e8c\u5143\u7ec4\uff0c\u8868\u793a\u5728\u4e0d\u540c\u7ef4\u5ea6\u4e0a\u7684\u6b65\u957f\uff0c",(0,r.jsx)(n.code,{children:"Offset"})," \u5219\u662f\u4e00\u4e2a\u5173\u4e8e ",(0,r.jsx)(n.code,{children:"parallel_id"})," \u548c ",(0,r.jsx)(n.code,{children:"loop_id"})," \u7684\u4eff\u5c04\u51fd\u6570\uff0c\u7528\u4e8e\u8868\u793a\u6bcf\u4e00\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u5728\u6bcf\u4e2a\u5faa\u73af\u4e2d\u6240\u5904\u7406\u7684\u5185\u5b58\u5207\u7247\u7684\u7684\u9996\u5730\u5740\u76f8\u5bf9\u4e8e\u5176\u6240\u57fa\u4e8e\u7684iPointer\u7684\u504f\u79fb\uff0c\u7528\u4e8e\u8868\u793a\u8f93\u5165\u8f93\u51fa\u6570\u636e\u548c\u4e2d\u95f4\u53d8\u91cf\u5728\u4e0d\u540c\u5e76\u884c\u5355\u5143\u548c\u5faa\u73af\u4e0a\u7684\u5212\u5206\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{children:"iPointer ::= iPointer(ID, Hierarchy, DType, Size)\nID ::= Int\nHierarchy ::= reg | sram | dram\nDType ::= fp64 | fp32 | fp16 ...\nSize ::= Int\n\niSlice ::= iSlice(iPointer, Offset, Shape, Stride)\nOffset ::= Affine Function of (parallel_id, loop_id)\nShape ::= (Int, Int)\nStride ::= (Int, Int)\n"})}),"\n",(0,r.jsxs)(n.p,{children:["\u4e3e\u4e2a\u4f8b\u5b50\uff0c\u5047\u8bbe\u4e00\u4e2a ",(0,r.jsx)(n.strong,{children:"iGraph"})," \u88ab\u7528\u4e8e\u8868\u793a\u4e00\u4e2a\u5f20\u91cf\u52a0\u6cd5\u7b97\u5b50\uff0c\u8f93\u5165\u5f20\u91cf\u7684\u5f62\u72b6\u4e3a",(0,r.jsx)(n.code,{children:"(8\uff0c 1024)"}),"\uff0c\u5e76\u4e14 ",(0,r.jsx)(n.code,{children:"(numParallel, numLoop) = (64, 2)"}),"\uff0c\u5219\u8fd9\u4e2a\u8f93\u5165\u5f20\u91cf\u53ef\u4ee5\u7528\u4e00\u4e2a ",(0,r.jsx)(n.strong,{children:"iPointer"})," \u63cf\u8ff0\uff1a"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"a = iPointer(0, dram, fp32, 8192)\n"})}),"\n",(0,r.jsxs)(n.p,{children:["\u6211\u4eec\u5728\u5404\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u95f4\u7684\u5404\u4e2a\u5faa\u73af\u95f4\u5747\u5300\u5206\u914d\u6240\u9700\u8981\u5904\u7406\u7684\u8f93\u5165\u6570\u636e\uff0c\u5219\u6570\u636e\u7684\u5212\u5206\u53ef\u4ee5\u7528 ",(0,r.jsx)(n.strong,{children:"iSlice"})," \u63cf\u8ff0\uff1a"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"aSlice = iSlice(a, [&](pid, lid) { return 4096 * lid + 128 * pid; }, (2, 32), (32, 1))\n"})}),"\n",(0,r.jsxs)(n.p,{children:["\u5047\u8bbe\u6211\u4eec\u5728\u7b97\u5b50\u5b9e\u73b0\u4e2d\uff0c\u9700\u8981\u5148\u5c06\u6570\u636e\u8bfb\u5165\u5bc4\u5b58\u5668\u4e2d\u518d\u8fdb\u884c\u8ba1\u7b97\uff0c\u5219\u53ef\u4ee5\u6309\u7167\u5982\u4e0b\u63cf\u8ff0\u8fd9\u4e9b\u5bc4\u5b58\u5668\uff0c\u53ef\u4ee5\u6ce8\u610f\u5230\u5bc4\u5b58\u5668\u662f\u6bcf\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u6240\u72ec\u5360\u7684\uff0c\u6240\u4ee5 ",(0,r.jsx)(n.code,{children:"b"})," \u4f4d\u4e8e ",(0,r.jsx)(n.code,{children:"reg"})," \u4e0a\uff0c\u5e76\u4e14\u5927\u5c0f\u4e3a64\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"b = iPointer(1, reg, fp32, 64)\nbSlice = iPointer(b, [&](pid, lid) { return 0; }, (2, 32), (32, 1))\n"})}),"\n",(0,r.jsx)(n.h3,{id:"iop",children:"iOp"}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.strong,{children:"iOp"})," \u662f\u5bf9\u786c\u4ef6\u6307\u4ee4\u7684\u62bd\u8c61\uff0c\u5176\u8f93\u5165\u8f93\u51fa\u90fd\u662f\u4e0a\u9762\u5b9a\u4e49\u7684 ",(0,r.jsx)(n.strong,{children:"iSlice"}),"\uff0c\u8868\u793a\u5728\u5185\u5b58\u5207\u7247\u4e0a\u7684\u786c\u4ef6\u6307\u4ee4\u64cd\u4f5c\u3002\u4f5c\u4e3a\u5bf9\u786c\u4ef6\u6307\u4ee4\u7684\u62bd\u8c61\uff0c\u4e00\u4e2a ",(0,r.jsx)(n.strong,{children:"iOp"})," \u901a\u5e38\u53ef\u4ee5\u5728\u4e00\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u4e0a\u7531\u4e00\u6761\u786c\u4ef6\u6307\u4ee4\u6216\u8005\u4e00\u4e2a\u786c\u4ef6\u6307\u4ee4\u5e8f\u5217\u5b8c\u6210\u3002",(0,r.jsx)(n.strong,{children:"iOp"})," \u5206\u4e3a\u8bbf\u5b58iOp\u548c\u8ba1\u7b97iOp\uff0c\u5206\u522b\u7528\u6765\u8868\u793a\u8bbf\u5b58\u64cd\u4f5c\u548c\u8ba1\u7b97\u3002"]}),"\n",(0,r.jsx)(n.h4,{id:"\u8bbf\u5b58iop",children:"\u8bbf\u5b58iOp"}),"\n",(0,r.jsx)(n.p,{children:"\u8bbf\u5b58iOp\u5206\u4e3a\u4e24\u79cd\uff0c\u4e00\u79cd\u662f\u79fb\u52a8\uff08Move\uff09\uff0c\u53e6\u4e00\u79cd\u662f\u540c\u6b65\uff08Sync\uff09\u3002"}),"\n",(0,r.jsxs)(n.p,{children:["\u79fb\u52a8iOp\u8868\u793a\u5728\u4e0d\u540c\u7684\u5185\u5b58\u5c42\u7ea7\u95f4\u79fb\u52a8\u5185\u5b58\u5207\u7247\uff0c\u6216\u8005\u5728\u540c\u4e00\u5185\u5b58\u5c42\u7ea7\u5185\u5bf9\u5185\u5b58\u5207\u7247\u7684\u5e03\u5c40\u8fdb\u884c\u6539\u53d8\uff0c\u5bf9\u5e94\u786c\u4ef6\u6307\u4ee4\u4e0a\u7684\u5185\u5b58\u76f8\u5173\u6307\u4ee4\uff0c\u5176\u8bed\u4e49\u662f\u5c06\u4f4d\u4e8e ",(0,r.jsx)(n.code,{children:"from"})," \u4e0a\u7684\u5185\u5b58\u5207\u7247 ",(0,r.jsx)(n.code,{children:"s1"})," \u642c\u8fd0\u5230\u4f4d\u4e8e ",(0,r.jsx)(n.code,{children:"to"})," \u4e0a\u7684\u5185\u5b58\u5207\u7247 ",(0,r.jsx)(n.code,{children:"t1"})," \u4e0a\uff0c\u5e76\u4e14\u642c\u8fd0\u7684\u6570\u636e\u7c7b\u578b\u4e3a ",(0,r.jsx)(n.code,{children:"dtype"}),"\u3002\u5b9e\u9645\u4e0a\uff0c\u7531\u4e8e ",(0,r.jsx)(n.code,{children:"s1"})," \u548c ",(0,r.jsx)(n.code,{children:"t1"})," \u4f5c\u4e3a\u5185\u5b58\u5207\u7247\u4e0d\u4ec5\u5305\u542b\u5176\u57fa\u4e8e\u7684\u6307\u9488\u3001\u5730\u5740\u504f\u79fb\u3001\u5f62\u72b6\u548c\u6b65\u957f\u7b49\u4fe1\u606f\uff0c\u4e5f\u5305\u542b\u4e86\u8be5\u5185\u5b58\u5207\u7247\u6240\u5728\u5c42\u7ea7\u548c\u5185\u90e8\u7684\u6570\u636e\u7c7b\u578b\uff0c\u56e0\u6b64 ",(0,r.jsx)(n.code,{children:"move"})," \u6307\u4ee4\u4e2d\u7684\u5c5e\u6027\u5747\u53ef\u4ee5\u4ece\u8f93\u5165\u8f93\u51fa\u4e2d\u63a8\u5bfc\u51fa\u6765\uff0c\u8fd9\u91cc\u5c06\u5176\u663e\u5f0f\u5b9a\u4e49\u51fa\u6765\u662f\u4e3a\u4e86\u6613\u8bfb\u6027\u3002\u5982\u65e0\u7279\u6b8a\u8bf4\u660e\uff0c\u4e0b\u9762\u4ecb\u7ecd\u7684\u5176\u4ed6 ",(0,r.jsx)(n.strong,{children:"iOp"})," \u5b9a\u4e49\u4e5f\u9075\u5faa\u8fd9\u4e00\u7ea6\u5b9a\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"move.from.to.dtype t1, s1\n\n.from = {.dram, .sram, .reg}\n.to = {.dram, .sram, .reg}\n.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}\n\n// examples\nmove.dram.sram.fp32 t1, s1\nmove.sram.reg.fp16 t1, s1\n"})}),"\n",(0,r.jsxs)(n.p,{children:["\u540c\u6b65iOp\u5219\u8868\u793a\u9700\u8981\u5728\u67d0\u4e2a\u5185\u5b58\u5c42\u7ea7\u4e0a\u5bf9\u76f8\u5e94\u7684\u5e76\u884c\u7ed3\u6784\u505a\u540c\u6b65\uff0c\u4ece\u800c\u4fdd\u8bc1\u6570\u636e\u4f9d\u8d56\u7684\u6b63\u786e\u6027\u3002\u53ef\u4ee5\u6ce8\u610f\u5230 ",(0,r.jsx)(n.code,{children:"sync"})," \u6307\u4ee4\u8981\u6c42 ",(0,r.jsx)(n.code,{children:"iPointer(t1) = iPointer(s1)"}),"\uff0c\u8fd9\u662f\u56e0\u4e3a\u53ea\u6709\u5f53\u7b97\u5b50\u5b9e\u73b0\u4e2d\u5199\u5165\u548c\u8bfb\u51fa\u7684\u5185\u5b58\u662f\u91cd\u53e0\u7684\u65f6\u5019\u624d\u9700\u8981\u540c\u6b65\u3002\u4f8b\u5982\uff0c",(0,r.jsx)(n.code,{children:"s1"})," \u548c ",(0,r.jsx)(n.code,{children:"t1"})," \u90fd\u5bf9\u5e94\u4e8e\u540c\u4e00\u4e2a ",(0,r.jsx)(n.code,{children:"dram"})," \u4e0a\u7684iPointer\uff0c\u4e14\u6709\u76f8\u540c\u7684\u5f62\u72b6\u548c\u6b65\u957f\uff0c\u4f46\u662f\u4e8c\u8005\u7684\u5730\u5740\u504f\u79fb\u4e0d\u540c\uff0c\u8fd9\u5c31\u610f\u5473\u7740\u4e00\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u6240\u8bfb\u53d6\u7684\u5207\u7247\u53ef\u80fd\u662f\u7531\u522b\u7684\u5355\u5143\u5199\u5165\u7684\uff0c\u56e0\u6b64\u9700\u8981\u8fdb\u884c\u540c\u6b65\u624d\u80fd\u4fdd\u8bc1\u6570\u636e\u4f9d\u8d56\u7684\u6b63\u786e\u6027\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"sync.scope t1, s1\n\n.scope = {.dram, .sram, .reg}\ns.t. iPointer(t1) = iPointer(s1)\n\n// examples\nsync.sram t1, s1\n"})}),"\n",(0,r.jsx)(n.h4,{id:"\u8ba1\u7b97iop",children:"\u8ba1\u7b97iOp"}),"\n",(0,r.jsxs)(n.p,{children:["\u8ba1\u7b97iOp\u6709\u51e0\u79cd\u7c7b\u578b\uff0c\u6709 ",(0,r.jsx)(n.code,{children:"unary"}),"\u3001",(0,r.jsx)(n.code,{children:"binary"}),"\u3001",(0,r.jsx)(n.code,{children:"broadcast"}),"\u3001",(0,r.jsx)(n.code,{children:"reduce"})," \u548c ",(0,r.jsx)(n.code,{children:"identity"}),"\u3002",(0,r.jsx)(n.code,{children:"unary"})," \u4e3a\u5355\u76ee\u8fd0\u7b97\uff0c\u8bed\u6cd5\u5982\u4e0b\uff0c\u5176\u4e2d ",(0,r.jsx)(n.code,{children:"name"})," \u8868\u793a\u8fd0\u7b97\u7c7b\u578b\uff0c",(0,r.jsx)(n.code,{children:"dtype"})," \u8868\u793a\u6570\u636e\u7c7b\u578b\uff0c",(0,r.jsx)(n.code,{children:"params"})," \u4e3a\u53ef\u9009\u53c2\u6570\uff0c\u7528\u4e8e ",(0,r.jsx)(n.code,{children:"muls"}),"\u3001",(0,r.jsx)(n.code,{children:"leaky_relu"})," \u7b49\u6307\u4ee4\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"unary.name.dtype t1, s1, [params]\n\n.name = {.muls, .adds, .divs, .subs, .exp, .log, .sin, .cos, .relu, ...}\n.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}\n\n// examples\nunary.muls.fp32 t1, s1 [2.0f, ]\nunary.exp.fp32 t1, s1\n"})}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.code,{children:"binary"})," \u4e3a\u53cc\u76ee\u8fd0\u7b97\uff0c\u5176\u5b9a\u4e49\u548c ",(0,r.jsx)(n.code,{children:"unary"})," \u7c7b\u4f3c\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"binary.name.dtype t1, s1, s2\n\n.name = {.mul, .add, .div, .sub, ...}\n.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}\n\n// examples\nbinary.mul.fp32 t1, s1, s2\nbinary.sub.fp32 t1, s1, s2\n"})}),"\n",(0,r.jsxs)(n.p,{children:[(0,r.jsx)(n.code,{children:"broadcast"})," \u548c ",(0,r.jsx)(n.code,{children:"reduce"})," \u8868\u793a\u5e7f\u64ad\u548c\u89c4\u7ea6iOp\uff0c\u5176\u4e2d ",(0,r.jsx)(n.code,{children:"dtype"})," \u8868\u793a\u6570\u636e\u7c7b\u578b\uff0c",(0,r.jsx)(n.code,{children:"axis"})," \u8868\u793a\u5728\u8fdb\u884c\u89c4\u7ea6\u6216\u8005\u5e7f\u64ad\u7684\u8f74\u3002\u800c ",(0,r.jsx)(n.code,{children:"scope"})," \u8868\u793a\u5728\u4ec0\u4e48\u8303\u56f4\u5185\u8fdb\u884c\u89c4\u7ea6\u6216\u8005\u5e7f\u64ad\uff0c\u5176\u4e2d ",(0,r.jsx)(n.code,{children:"unit"})," \u8868\u793a\u5728\u6bcf\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u5185\uff0c\u800c ",(0,r.jsx)(n.code,{children:"group"})," \u8868\u793a\u5728\u540c\u4e00\u5e76\u884c\u7ec4\u5185\u7684\u591a\u4e2a\u6700\u5c0f\u5e76\u884c\u5355\u5143\u4e4b\u95f4\uff0c",(0,r.jsx)(n.code,{children:"op"})," \u8868\u793a\u7528\u4e8e\u89c4\u7ea6\u7684\u8ba1\u7b97\u7c7b\u578b\u3002\u5bf9\u4e8e ",(0,r.jsx)(n.code,{children:"group"})," \u5c42\u7ea7\u4e0a\u7684\u89c4\u7ea6\u6216\u8005\u5e7f\u64ad\u6d89\u53ca\u5230\u4e0d\u540c\u5e76\u884c\u5355\u5143\uff0c\u56e0\u6b64\u9700\u8981\u5728 ",(0,r.jsx)(n.code,{children:"sram"})," \u4e0a\u901a\u4fe1\uff0c\u6240\u4ee5\u9700\u8981\u5728 ",(0,r.jsx)(n.code,{children:"params"})," \u4e2d\u6307\u5b9a\u4e00\u4e2a\u4f4d\u4e8e ",(0,r.jsx)(n.code,{children:"sram"})," \u4e0a\u7684iSlice\u4f5c\u4e3abuffer\u6765\u8f85\u52a9\uff0c\u53e6\u5916\u53c2\u6570groupSize\u6307\u5b9a\u7684\u8fdb\u884c\u5e7f\u64ad\u6216\u8005\u89c4\u7ea6\u7684\u7ec4\u7684\u5927\u5c0f\uff0c\u9700\u8981\u6ee1\u8db3 ",(0,r.jsx)(n.code,{children:"numUnit % groupSize == 0"}),"\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"broadcast.axis.scope.dtype t1, s1, [params]\n\n.axis = {.col, .row}\n.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}\n.scope = {.unit, .group}\n\n// example\nbroadcast.row.unit.fp32 t1, s1 // Shape(s1) = (4, 1), Shape(t1) = (4, 128)\nbroadcast.col.unit.fp32 t1, s1 // Shape(s1) = (1, 128), Shape(t1) = (4, 128)\n\nbroadcast.row.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (4, 1), Shape(t1) = (4, 128), numUnit = 8\nbroadcast.col.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (1, 128), Shape(t1) = (4, 128), numUnit = 8\n"})}),"\n",(0,r.jsxs)(n.p,{children:["\u5bf9\u4e8e ",(0,r.jsx)(n.code,{children:"group"})," \u5c42\u7ea7\u4e0a\u7684\u89c4\u7ea6\uff0c\u5176\u6700\u7ec8\u89c4\u7ea6\u7684\u7ed3\u679c\u4f1a\u4fdd\u5b58\u5728 ",(0,r.jsx)(n.code,{children:"parallel_id % groupSize == 0"})," \u7684\u6700\u5c0f\u5e76\u884c\u5355\u5143\u4e0a\uff1b\u800c\u5bf9\u4e8e ",(0,r.jsx)(n.code,{children:"group"})," \u5c42\u7ea7\u4e0a\u7684\u5e7f\u64ad\uff0c\u5176\u5e7f\u64ad\u7684\u503c\u6765\u81ea ",(0,r.jsx)(n.code,{children:"parallel_id % groupSize == 0"})," \u7684\u6700\u5c0f\u5e76\u884c\u5355\u5143\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"reduce.op.axis.scope.dtype t1, s1, [params]\n\n.op = {.max, .min, .add, .mul}\n.axis = {.col, .row}\n.dtype = {.fp64, .fp32, .fp16, .bf16, .fp8}\n.scope = {.unit, .group}\n\n// examples\nreduce.add.row.unit.fp32 t1, s1 // Shape(s1) = (4, 128), Shape(t1) = (4, 1)\nreduce.add.col.unit.fp32 t1, s1 // Shape(s1) = (4, 128), Shape(t1) = (1, 128)\n\nreduce.add.row.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (4, 128), Shape(t1) = (1, 128), numUnit = 8\nreduce.add.col.group.fp32 t1, s1, [buffer=b1, groupSize=4] // Shape(s1) = (4, 128), Shape(t1) = (1, 128), numUnit = 8\n"})}),"\n",(0,r.jsxs)(n.p,{children:["\u9664\u4e86\u4ee5\u4e0a\u8fd9\u4e9biOp\u4e4b\u5916\uff0c\u8fd8\u6709\u4e00\u7c7b\u4e0d\u8868\u793a\u5177\u4f53\u8ba1\u7b97\u4f46\u662f\u5bf9Unified IR\u7684\u8868\u8fbe\u6027\u5341\u5206\u91cd\u8981\u7684iOp\uff0c\u79f0\u4e3a ",(0,r.jsx)(n.code,{children:"identity"}),"\u3002\u987e\u540d\u601d\u4e49\uff0c\u5b83\u8868\u793a\u8f93\u5165\u8f93\u51fa\u7684\u4e24\u7ec4iSlice\u5728\u7269\u7406\u5185\u5b58\u4e0a\u662f\u540c\u4e00\u5757\uff0c\u53ef\u4ee5\u7528\u4e8e\u5728iGraph\u4e2d\u8868\u8fbeReshape\u3001Split\u3001Concat\u7b49\u903b\u8f91\u3002",(0,r.jsx)(n.code,{children:"identity"})," \u6307\u4ee4\u8981\u6c42\u8f93\u5165\u8f93\u51fa\u7684 ",(0,r.jsx)(n.code,{children:"iSlice"})," \u8868\u793a\u540c\u4e00\u5757\u7684\u7269\u7406\u5185\u5b58\u3002"]}),"\n",(0,r.jsx)(n.pre,{children:(0,r.jsx)(n.code,{className:"language-c",children:"identity [ts], [ss]\n\n// example\nidentity t1, s1 // Shape(t1) = (4, 64), Shape(s1) = (2, 128)\nidentity t1, [s1, s2] // Shape(t1) = (4, 64), Shape(s1) = Shape(s2) = (4, 32)\nidentity [t1, t2], s1 // Shape(t1) = Shape(t2) = (4, 32), Shape(s1) = (4, 64)\n"})})]})}function p(e={}){const{wrapper:n}={...(0,s.R)(),...e.components};return n?(0,r.jsx)(n,{...e,children:(0,r.jsx)(t,{...e})}):t(e)}},8453:(e,n,i)=>{i.d(n,{R:()=>c,x:()=>l});var r=i(6540);const s={},d=r.createContext(s);function c(e){const n=r.useContext(d);return r.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(s):e.components||s:c(e.components),r.createElement(d.Provider,{value:n},e.children)}}}]);