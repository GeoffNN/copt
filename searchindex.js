Search.setIndex({docnames:["auto_examples/frank_wolfe/plot_fw_stepsize","auto_examples/frank_wolfe/plot_fw_vertex_overlap","auto_examples/frank_wolfe/plot_stepsize","auto_examples/frank_wolfe/plot_vertex_overlap","auto_examples/frank_wolfe/sg_execution_times","auto_examples/index","auto_examples/plot_accelerated","auto_examples/plot_group_lasso","auto_examples/plot_jax_copt","auto_examples/plot_saga_vs_svrg","auto_examples/proximal_splitting/plot_overlapping_group_lasso","auto_examples/proximal_splitting/plot_sparse_nuclear_norm","auto_examples/proximal_splitting/plot_tv_deblurring","auto_examples/proximal_splitting/sg_execution_times","auto_examples/sg_execution_times","datasets","frank_wolfe","generated/copt.datasets.load_covtype","generated/copt.datasets.load_img1","generated/copt.datasets.load_rcv1","generated/copt.datasets.load_url","generated/copt.minimize_frank_wolfe","generated/copt.minimize_pairwise_frank_wolfe","generated/copt.minimize_primal_dual","generated/copt.minimize_proximal_gradient","generated/copt.minimize_saga","generated/copt.minimize_svrg","generated/copt.minimize_three_split","generated/copt.minimize_vrtos","generated/copt.utils.GroupL1","generated/copt.utils.HuberLoss","generated/copt.utils.L1Ball","generated/copt.utils.L1Norm","generated/copt.utils.LogLoss","generated/copt.utils.SquareLoss","generated/copt.utils.TotalVariation2D","generated/copt.utils.TraceBall","generated/copt.utils.TraceNorm","incremental","index","loss_functions","proximal_gradient","proximal_splitting"],envversion:{"sphinx.domains.c":1,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":1,"sphinx.domains.javascript":1,"sphinx.domains.math":2,"sphinx.domains.python":1,"sphinx.domains.rst":1,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":1,sphinx:56},filenames:["auto_examples/frank_wolfe/plot_fw_stepsize.rst","auto_examples/frank_wolfe/plot_fw_vertex_overlap.rst","auto_examples/frank_wolfe/plot_stepsize.rst","auto_examples/frank_wolfe/plot_vertex_overlap.rst","auto_examples/frank_wolfe/sg_execution_times.rst","auto_examples/index.rst","auto_examples/plot_accelerated.rst","auto_examples/plot_group_lasso.rst","auto_examples/plot_jax_copt.rst","auto_examples/plot_saga_vs_svrg.rst","auto_examples/proximal_splitting/plot_overlapping_group_lasso.rst","auto_examples/proximal_splitting/plot_sparse_nuclear_norm.rst","auto_examples/proximal_splitting/plot_tv_deblurring.rst","auto_examples/proximal_splitting/sg_execution_times.rst","auto_examples/sg_execution_times.rst","datasets.rst","frank_wolfe.rst","generated/copt.datasets.load_covtype.rst","generated/copt.datasets.load_img1.rst","generated/copt.datasets.load_rcv1.rst","generated/copt.datasets.load_url.rst","generated/copt.minimize_frank_wolfe.rst","generated/copt.minimize_pairwise_frank_wolfe.rst","generated/copt.minimize_primal_dual.rst","generated/copt.minimize_proximal_gradient.rst","generated/copt.minimize_saga.rst","generated/copt.minimize_svrg.rst","generated/copt.minimize_three_split.rst","generated/copt.minimize_vrtos.rst","generated/copt.utils.GroupL1.rst","generated/copt.utils.HuberLoss.rst","generated/copt.utils.L1Ball.rst","generated/copt.utils.L1Norm.rst","generated/copt.utils.LogLoss.rst","generated/copt.utils.SquareLoss.rst","generated/copt.utils.TotalVariation2D.rst","generated/copt.utils.TraceBall.rst","generated/copt.utils.TraceNorm.rst","incremental.rst","index.rst","loss_functions.rst","proximal_gradient.rst","proximal_splitting.rst"],objects:{"copt.datasets":{load_covtype:[17,0,1,""],load_img1:[18,0,1,""],load_rcv1:[19,0,1,""],load_url:[20,0,1,""]},"copt.utils":{GroupL1:[29,1,1,""],HuberLoss:[30,1,1,""],L1Ball:[31,1,1,""],L1Norm:[32,1,1,""],LogLoss:[33,1,1,""],SquareLoss:[34,1,1,""],TotalVariation2D:[35,1,1,""],TraceBall:[36,1,1,""],TraceNorm:[37,1,1,""]},"copt.utils.GroupL1":{__init__:[29,2,1,""]},"copt.utils.HuberLoss":{__init__:[30,2,1,""]},"copt.utils.L1Ball":{__init__:[31,2,1,""],lmo:[31,2,1,""]},"copt.utils.L1Norm":{__init__:[32,2,1,""]},"copt.utils.LogLoss":{Hessian:[33,2,1,""],__init__:[33,2,1,""]},"copt.utils.SquareLoss":{__init__:[34,2,1,""]},"copt.utils.TotalVariation2D":{__init__:[35,2,1,""]},"copt.utils.TraceBall":{__init__:[36,2,1,""]},"copt.utils.TraceNorm":{__init__:[37,2,1,""]},copt:{minimize_frank_wolfe:[21,0,1,""],minimize_pairwise_frank_wolfe:[22,0,1,""],minimize_primal_dual:[23,0,1,""],minimize_proximal_gradient:[24,0,1,""],minimize_saga:[25,0,1,""],minimize_svrg:[26,0,1,""],minimize_three_split:[27,0,1,""],minimize_vrtos:[28,0,1,""]}},objnames:{"0":["py","function","Python function"],"1":["py","class","Python class"],"2":["py","method","Python method"]},objtypes:{"0":"py:function","1":"py:class","2":"py:method"},terms:{"00it":[3,10,11,12],"01e":[7,9,10,11,12],"01it":[1,7,10,11,12],"02e":[7,9,10,11,12],"02it":[3,7,10,11,12],"03e":[7,9,10,11,12],"03it":[1,7,10,11,12],"04e":[7,9,10,11,12],"04it":[10,11,12],"05e":[7,10,11,12],"05it":[1,10,11,12],"06e":[7,9,10,11,12],"06it":[10,11,12],"07e":[1,3,7,10,11,12],"07it":[7,10,11,12],"08e":[1,3,7,9,10,11,12],"08it":[1,10,11,12],"09e":[1,3,7,9,10,11,12],"09it":[1,7,10,11,12],"10it":[1,3,10,11,12],"11e":[7,10,11,12],"11it":[1,3,7,10,11,12],"12e":[7,10,11,12],"12it":[3,7,10,11,12],"13e":[7,10,11,12],"13it":[1,3,10,11,12],"14e":[7,10,11,12],"14it":[1,3,7,10,11,12],"15e":[7,10,11,12],"15it":[1,3,7,10,11,12],"16e":[7,10,11,12],"16it":[1,3,7,10,11,12],"17e":[7,9,10,11,12],"17it":[1,10,11,12],"18e":[7,9,10,11,12],"18it":[1,3,7,10,11,12],"19e":[7,9,10,11,12],"19it":[1,3,7,10,11,12],"20it":[3,10,11,12],"21e":[7,9,10,11,12],"21it":[1,3,7,10,11,12],"22e":[7,10,11,12],"22it":[1,3,7,10,11,12],"23e":[7,9,10,11,12],"23it":[1,7,10,11,12],"24e":[7,10,11,12],"24it":[1,3,7,10,11,12],"25e":[7,9,10,11,12],"25it":[1,3,10,11,12],"26e":[7,10,11,12],"26it":[1,7,10,11,12],"27e":[7,9,10,11,12],"27it":[1,3,7,10,11,12],"28e":[7,10,11,12],"28it":[1,3,7,10,11,12],"29e":[7,10,11,12],"29it":[1,3,10,11,12],"30it":[1,3,7,10,12],"31e":[7,9,10,11,12],"31it":[1,3,7,10,11,12],"32e":[7,9,10,11,12],"32it":[3,7,10,11,12],"33e":[7,9,10,11,12],"33it":[3,7,10,11,12],"34e":[7,10,11,12],"34it":[3,10,11,12],"35e":[7,9,10,11,12],"35it":[3,7,10,11,12],"35th":27,"36e":[7,9,10,11,12],"36it":[3,10,11,12],"37e":[7,10,11,12],"37it":[1,3,7,9,10,11,12],"38e":[7,9,10,11,12],"38it":[3,10,11,12],"39e":[7,10,11,12],"39it":[3,7,10,12],"40it":[3,10,11,12],"41e":[7,10,11,12],"41it":[3,7,10,11,12],"42e":[7,9,10,11,12],"42it":[3,7,10,11,12],"43e":[7,10,11,12],"43it":[3,10,11,12],"44e":[7,10,11,12],"44it":[7,10,11,12],"45e":[7,10,11,12],"45it":[7,10,11,12],"46e":[7,9,10,11,12],"46it":[3,7,10,11,12],"47e":[7,10,11,12],"47it":[1,3,10,11,12],"48e":[7,10,11,12],"48it":[7,9,10,11,12],"49e":[7,10,11,12],"49it":[1,10,11,12],"50it":[3,10,11,12],"51e":[7,9,10,11,12],"51it":[3,7,10,11,12],"52e":[7,10,11,12],"52it":[1,7,10,11,12],"53e":[7,9,10,11,12],"53it":[10,11,12],"54e":[7,9,10,11,12],"54it":[3,10,11,12],"55e":[7,9,10,11,12],"55it":[7,10,11,12],"56e":[7,10,11,12],"56it":[7,9,10,12],"57e":[7,10,11,12],"57it":[7,10,11,12],"58e":[7,9,10,11,12],"58it":[1,7,10,11,12],"59e":[7,10,11,12],"59it":[1,7,10,12],"60it":[1,7,10,11,12],"61e":[7,9,10,11,12],"61it":[1,7,10,11,12],"62e":[7,10,11,12],"62it":[1,7,10,11,12],"63e":[7,10,11,12],"63it":[3,10,11,12],"64e":[7,10,11,12],"64it":[1,7,10,11,12],"65e":[7,9,10,11,12],"65it":[10,12],"66e":[7,9,10,11,12],"66it":[1,3,7,10,12],"67e":[7,9,10,11,12],"67it":[1,3,10,11,12],"68e":[7,10,11,12],"68it":[1,7,10,11,12],"69e":[7,9,10,11,12],"69it":[1,7,10,11,12],"70it":[1,3,7,10,11,12],"71e":[7,9,10,11,12],"71it":[3,10,11,12],"72e":[7,10,11,12],"72it":[1,3,10,11,12],"73e":[7,10,11,12],"73it":[1,7,10,11,12],"74e":[7,10,11,12],"74it":[7,10,11,12],"75e":[7,10,11,12],"75it":[1,3,7,10,11,12],"76e":[7,9,10,11,12],"76it":[10,11,12],"77e":[7,9,10,11,12],"77it":[1,3,7,9,10,11,12],"78e":[7,9,10,11,12],"78it":[1,10,11,12],"79e":[7,9,10,11,12],"79it":[7,10,11,12],"80it":[7,10,11,12],"81e":[7,9,10,11,12],"81it":[10,11,12],"82e":[7,9,10,11,12],"82it":[1,3,7,9,10,11,12],"83e":[1,3,7,10,11,12],"83it":[1,7,10,11,12],"84e":[1,3,7,10,11,12],"84it":[1,3,7,10,11,12],"85e":[1,3,7,9,10,11,12],"85it":[7,10,11,12],"86e":[1,3,7,9,10,11,12],"86it":[7,10,11,12],"87e":[1,3,7,9,10,11,12],"87it":[3,7,10,11,12],"88e":[1,3,7,9,10,11,12],"88it":[10,11,12],"89e":[1,3,7,10,11,12],"89it":[3,7,10,11,12],"90it":[3,7,10,11,12],"91e":[1,3,7,9,10,11,12],"91it":[7,10,11,12],"92e":[1,3,7,9,10,11,12],"92it":[3,7,10,11,12],"93e":[1,3,7,9,10,11,12],"93it":[7,9,10,11,12],"94e":[1,3,7,9,10,11,12],"94it":[1,10,11,12],"95e":[1,3,7,10,11,12],"95it":[1,3,7,9,10,11,12],"96e":[1,3,7,9,10,11,12],"96it":[3,7,10,12],"97e":[1,3,7,9,10,11,12],"97it":[1,7,10,12],"98e":[1,3,7,9,10,11,12],"98it":[10,12],"99e":[7,10,11,12],"99it":[1,7,10,11,12],"boolean":[21,23,24,25,26,27,28],"break":[25,26],"case":[9,16],"class":[29,30,31,32,33,34,35,36,37],"default":16,"float":[12,21,24,25,26,27,28,29,32],"function":[0,1,2,3,6,7,8,9,16,21,23,24,25,26,27,28,31,33],"import":[0,1,2,3,6,7,8,9,10,11,12,21,23,24,25,26,27,28],"int":[11,21,23,24,25,26,27,28],"return":[7,8,10,11,12,16,17,19,20,21,23,24,25,26,27,28,33],"true":[1,3,6,7,10,11,12,19,20,23,25,26,27,28],For:16,TOS:[10,11,12],The:[6,7,9,16,21,23,24,25,26,27,28,33],Then:16,These:39,Use:[6,9],With:27,__doc__:11,__init__:[29,30,31,32,33,34,35,36,37],_base:[6,9],a_i:[25,26,28],aaron:26,abov:[16,21],abs:[11,12],absolut:[31,32],acceler:[5,14,24],accept:[8,21,23],access:[16,24,25,40],account:9,accur:[29,30,31,32,33,34,35,36,37],achiev:[7,10,11,12],acquir:16,adapt:[0,1,2,3,10,12,16,21,24,27],adaptive2:[0,2,21],adaptive3:[0,1,2,3],adato:12,advanc:[16,25,26],after:10,agg:[0,1,2,3,6,7,8,9,10,11,12],aka:[36,37],algorithkm:16,algorithm:[1,3,9,21,23,24,25,26,27,28],align:[],all:5,all_beta:[7,10,11,12],all_trace_l:[7,10,11,12],all_trace_ls_tim:[10,11,12],all_trace_nol:[7,10,11,12],all_trace_nols_tim:[10,11,12],all_trace_pdhg:[10,11,12],all_trace_pdhg_nol:[10,11],all_trace_pdhg_nols_tim:[10,11],all_trace_pdhg_tim:[10,11,12],along:[],alpha:[16,23,25,26,27,28,29,30,31,32,33,34,35,36,37],alreadi:[39,40],also:16,alt:[],altern:[6,9,16,39],although:16,alwai:[],amir:24,anaconda3:[0,1,2,3,6,7,8,9,10,11,12],analysi:27,ani:16,antonin:23,api:[22,25,39],append:[0,1,2,3,7,10,11,12],appendix:10,applic:[16,23,24,27],arang:[7,9,10],arg:27,argmin_:[16,26,28],argument:[6,9,16,21,23,24,27],armin:[16,21],arrai:[7,10,11,12,17,19,20,21,23,24,25,26,27,28],art:39,arxiv:[16,21,28],askari:[16,21],assum:16,astyp:[11,12],asynchron:[26,28],attribut:[21,23,24,25,26,27,28,30,33,34,36,37],augment:[25,26],auto_exampl:14,auto_examples_frank_wolf:4,auto_examples_jupyt:5,auto_examples_proximal_split:13,auto_examples_python:5,averag:[25,26],axes:[1,3,6,9],axi:12,b_i:[25,26,28,33],bach:26,back:8,backend:[0,1,2,3,6,7,8,9,10,11,12],backtrack:[],backtracking_factor:[24,27],ball:[16,22,31],barrier:[25,26],base:24,bbox_to_anchor:[7,11],beck:24,been:[1,3],below:[16,21,25,26,28],beta:[7,10,11,12,23,26],between:[6,9,10],bianp:33,binari:[17,19,20],blck:11,block:[10,11,29],blur:12,blur_matrix:12,bold:[6,9],boldsymbol:16,bool:[19,20,25,26,28],both:8,bottom:[6,7,9,10,11,12],broadli:16,callabl:[21,23,24,27,33],callback:[0,1,2,3,6,7,8,9,10,11,12,21,22,23,24,25,26,27,28],can:[8,16,25,26,27,28,39],cannot:[0,1,2,3,6,7,8,9,10,11,12],casotto:28,categor:39,caus:[21,23,24,25,26,27,28],cb_adato:12,cb_apgd:6,cb_pdhg:[10,12],cb_pdhg_nol:10,cb_pgd:6,cb_saga:9,cb_svrg:9,cb_to:[7,10,11,12],cb_tosl:[7,10,11],cdot:34,center:[10,12],chambol:23,chang:[22,25],check:[1,3,19,20],chosen:16,cjlin:[17,19,20],classif:[0,2,17,19,20],click:[0,1,2,3,6,7,8,9,10,11,12],clip:[0,1,2,3],cmap:[11,12],code:[0,1,2,3,5,6,7,8,9,10,11,12],coeffici:[23,24,27],coincid:[1,3],com:39,combin:[5,14,16],come:33,command:39,commun:24,compact:16,comparison:[1,3,4,5,6,7,9,10,12,16,21],compart:16,competit:16,composit:[23,25,26],comput:[7,8,9,10,11,12],condat:23,condit:16,confer:27,constant:[16,21,32],constat:29,constrain:16,constraint:[0,2],construct:[6,7,8,9,16],contain:39,contrari:16,conveni:40,converg:[0,2,6,16,23,25,26,28],convex:[16,21,23,24,26],copt:[0,1,2,3,5,6,7,9,10,11,12,14,16],correl:[10,16],correspond:16,could:[1,3],cov:11,covtyp:[0,1,2,3,17],cpu:8,creat:10,criterion:[21,25,26,27,28],csie:[17,19,20],csr:[17,19,20],culo:[],current:[0,1,2,3,6,7,8,9,10,11,12,16,21,23,24,27],curv:16,damek:27,data:[7,8,10,11,12,25,26,28],dataset:[0,1,2,3,6,8,9,10,26],dataset_titl:[0,1,2,3],davi:27,debug:[25,26,28],decreas:21,def:[0,1,2,3,7,8,10,11,12],defazio:26,defin:[16,21,33,34,40],delta:30,demyanov:[16,21],densiti:7,depend:39,deploy:39,deprec:[6,9],deriv:[26,28,33,34],derivatives_logist:33,descent:[5,14,16,24,39],describ:[16,21,23,24,25,26,27,28],descript:[21,23,24,25,26,27,28],desir:[6,7],detail:21,determin:[1,3],dev:[6,7],develop:39,did:[6,7],diff:12,differ:[4,5,7,10,11,12,16,21,39],differenti:16,dimension:[16,35],direct:[4,5,16,21],doe:16,doesn:16,doi:[],domain:16,dot:[7,8,10,11,12,33],download:[0,1,2,3,5,6,7,8,9,10,11,12,17,19,20],draft:33,drawback:16,dt_prev:[1,3],dual:[12,23,39],dure:[1,3],each:[16,21,23,24,27],easi:39,easiest:39,edu:[17,19,20],element:[],elif:[1,3],ell_1:16,els:[1,3],emphasi:39,ensur:[],enter:7,enumer:[1,3,7,10,11,12],epsilon:11,eqref:[],equal:31,ergod:23,error:8,estim:[0,1,2,3,5,6,7,8,9,10,12,13,21,25,26,27,28,42],euclidean:34,evalu:[6,9,33,34],exampl:[0,1,2,3,6,7,8,9,10,11,12,21,24],exceed:21,exclus:16,execut:[1,3,4,13,14,21],exit:[21,23,24,25,26,27,28],experi:10,experiment:[22,25],explain:[],extra:[25,26,28],extrem:[1,3],eye:11,f_deriv:[25,26,28],f_grad:[0,1,2,3,6,7,8,10,11,12,21,22,23,24,27],fabian:[16,21,25,26,27,28],face:12,fall:8,fals:[6,7,10,11,12,21,23,24,26,27],fast:26,fastest:16,fatra:28,featur:[7,10,11,22],few:39,fig:[1,3,7,10],figlegend:[7,10,11,12],figsiz:[1,3],figur:[0,1,2,3,6,7,8,9,10,11,12],file:[4,13,14,19,20],first:[10,23],fit_transform:10,flag:[21,23,24,25,26,27,28],fmin:[6,7,9,10,11,12],follow:40,fontweight:[6,9],form:[16,21,23,24,25,26,27,28],format:8,found:[8,17,19,20],frac:[16,33,34],frameon:[7,10,11,12],franci:26,frank:[4,21,39],free:21,from:[1,3,7,8,10,11,12,23,24,25,26,27,28,33,39],fromarrai:12,full:[0,1,2,3,6,7,8,9,10,11,12,19],fw_iter:[],fw_trace:[0,2],g_prox:12,g_t:[0,2],galleri:[0,1,2,3,5,6,7,8,9,10,11,12],gamma:[12,16],gamma_0:[],gamma_t:[],gap:[0,2,21],gauthier:27,gcf:[7,10,11,12],gener:[0,1,2,3,5,6,7,8,9,10,11,12,16,25,26,28,39],geoffrei:[16,21],get_backend:[0,1,2,3,6,7,8,9,10,11,12],gidel:27,gisett:[0,1,2,3],git:39,github:39,given:[16,21,25,26,28],global:16,googl:[0,1,2,3,6,7,8,9,10,11,12],gpu:8,grad:[],gradient:[5,8,9,12,14,16,21,23,24,25,26,27,28,39],grai:12,gray_r:11,grid:[0,1,2,3,6,7,8,9,10,11,12],ground:7,ground_truth:[7,10],group:[5,13,14,24,29,42],groupl1:[7,10],guess:[21,23,24,27],gui:[0,1,2,3,6,7,8,9,10,11,12],h_lipschitz:[10,11,12,27],h_prox:12,has:[16,26],have:[1,3,16,24,25,26,28,33,39],help:[29,30,31,32,33,34,35,36,37],henc:16,here:[0,1,2,3,6,7,8,9,10,11,12],hessian:33,higher:[26,28],home:[0,1,2,3,6,7,8,9,10,11,12],how:[1,3,8],howev:40,html:[17,19,20,33],http:[17,19,20,33,39],huber:30,huberloss:11,hybrid:[12,23,39],icml:[16,21],illustr:16,imag:[12,18],img:12,immedi:21,implement:[7,9,16,21,26,39],impli:[1,3],improv:26,imshow:[11,12],includ:16,increment:26,indic:[21,23,24,25,26,27,28,31],inexact:16,infin:31,inform:[16,25,26],initi:[21,23,24,27,29,30,31,32,33,34,35,36,37],input:[21,33],insid:23,instal:39,instead:[6,9,16],integ:21,intern:27,interpol:[11,12],involv:23,ipynb:[0,1,2,3,6,7,8,9,10,11,12],iter:[0,1,2,3,7,8,9,10,11,12,16,21,23,24,27],its:[7,16,25,27],jaggi:[16,21],jax:[5,14],journal:23,julien:[16,25,26],jupyt:[0,1,2,3,5,6,7,8,9,10,11,12],keyword:[16,21],kilian:28,knowledg:16,known:[23,27],l1_ball:[0,1,2,3,8],l1ball:[0,1,2,3],l1norm:[8,9,11],l_t:[1,3],label:[0,1,2,3,6,9,10,17,19,20,33],lacost:[16,25,26],lambda:[7,8,9,10,11,12],langl:16,larg:39,larger:[1,3],lasso:[5,13,14,24,29,42],last:[1,3],latest:39,laurent:23,learn:27,least:10,leblond:[25,26],legend:[0,1,2,3,6,9],len:[9,10,11],leq:[16,33],less:31,level:[6,7,16,21,23,24,25,26,27,28],lib:[0,1,2,3,6,7,8,9,10,11,12],librari:[16,39],libsvm:[17,19,20],libsvmtool:[17,19,20],like:[16,21,22,23,24,25,26,27],linalg:[1,3,11,12],line:[7,10,11,16,27],line_search:[10,11,12,23,27],linear:[16,21,23,31],linearli:25,lipschitz:[0,1,2,3,6,7,10,11,12,16,21,22],lipschitzian:23,list:29,lmo:[0,1,2,3,16,21,22,31],lmo_act:22,load:[0,1,2,3,18],load_covtyp:[0,1,2,3],load_data:[0,1,2,3],load_gisett:[0,1,2,3],load_madelon:[0,1,2,3],load_npz:12,load_rcv1:[0,1,2,3],loc:[7,11],local:[0,1,2,3,6,7,8,9,10,11,12],log:[0,2,6,7,8,9,10,11,12,33],logist:[6,9,33],logloss:[0,1,2,3,6,9,10],logspac:10,loss:[7,8,10,11,12,25,29,30,33,34],low:[5,13,42],lower:[10,12],machin:27,madelon:[0,1,2,3],make:[1,3,10],make_regress:8,mani:[1,3],map:[25,26,28],marc:24,marker:[1,3,7,10,11,12],markers:[7,10,11,12],markeveri:[1,3,7,10,11,12],martin:[16,21],math:[],mathcal:16,mathemat:[23,39],matplotlib:[0,1,2,3,6,7,8,9,10,11,12],matplotlibdeprecationwarn:[6,9],matrix:[5,10,13,17,19,20,23,42],mattia:28,max:[10,12],max_it:[1,3,7,9,10,11,12,21,22,23,24,25,26,27,28,35],max_iter_backtrack:[24,27],max_iter_l:23,max_lipschitz:9,maximum:[21,23,24,25,26,27,28],maxit:[11,12],md5:[19,20],md5_check:[19,20],member:[25,26,28],memori:[0,1,2,3,6,7,8,9,10,11,12],messag:[21,23,24,25,26,27,28],method:[9,16,23,25,26,27,28,29,30,31,32,33,34,35,36,37,39],might:[1,3,25,26,28],min:[6,7,9,10,11,12],min_:31,minim:[16,21],minimize_frank_wolf:[0,1,2,3,16],minimize_pairwise_frank_wolf:16,minimize_primal_du:[10,12],minimize_proximal_gradi:[6,7,8],minimize_saga:9,minimize_svrg:9,minimize_three_split:[10,11,12],minimize_x:[23,24,25,27],minu:[7,10,11,12],minut:[0,1,2,3,6,7,8,9,10,11,12],misc:12,model:10,modular:39,more:[16,21],most:16,move:[],multipli:[9,11,29,32],multivariate_norm:11,n_col:[12,18],n_featur:[0,1,2,3,6,7,8,9,10,11,12,24],n_job:[26,28],n_row:[12,18],n_sampl:[0,1,2,3,6,7,8,9,10,11,12,25,26,28],nabla:16,ncol:[1,3,7,10,11,12],ndarrai:[23,25,26,28],nearest:[11,12],need:16,neg:16,negiar:[16,21],nesterov:6,net:33,neural:[16,25,26],newli:16,next:16,nip:[25,26],noisi:[12,16],non:[0,1,2,3,6,7,8,9,10,11,12,23,26,27,40],none:[0,1,2,3,7,10,11,12,21,22,23,24,25,26,27,28],nonsmooth:[25,26,28],nonzero:7,norm:[1,3,9,25,26,28,32,34,35,36,37],note:[16,21],notebook:[0,1,2,3,5,6,7,8,9,10,11,12],npz:12,nrow:[1,3],ntu:[17,19,20],nuclear:[36,37],number:[0,1,2,3,9,16,21,23,24,25,26,27,28],numpi:[0,1,2,3,6,7,8,9,10,11,12,17,19,20,39],obj_typ:[6,9],object:[7,8,9,10,11,12,16,21,23,24,25,26,27,28],oblivi:16,onli:[17,19,20,21,23,27],onp:8,onto:36,openopt:39,oper:[7,10,12,23,24,25,27,28,39,40],opt:[25,26,28,39],optim:[6,7,9,16,21,23,24,25,26,27,28],optimizeresult:[21,23,24,25,26,27,28],optimum:[7,10,11,12],option:[16,21,23,24,25,26,27,28],oracl:[16,21],order:23,org:[],origin:26,other:[21,23,24,25,26,27,28],otherwis:[0,1,2,3,31],out:[0,1,2,3,6,7,8,9,10,11,12],out_img:[7,10,11,12],output:[23,24,27],over:31,overlap:[4,5,13,16,21,42],own:40,packag:[0,1,2,3,6,7,8,9,10,11,12],pairwis:[1,3,22,39],parallel:[25,26],paramet:[7,10,11,12,16,19,20,21,23,24,25,26,27,28,29,32],parametr:25,partial_deriv:9,pass:[1,3,25,26,28],pdhg:10,pdhg_nol:10,pedregosa:[0,1,2,3,6,7,8,9,10,11,12,16,21,25,26,27,28],pen1:28,pen2:28,pen:12,penalti:29,perform:[16,27,33,39],pgd:7,pgd_l:7,pil:12,pip:39,plot:[0,1,2,3,6,7,8,9,10,11,12,25,26,28],plot_acceler:[6,14],plot_fw_steps:0,plot_fw_vertex_overlap:1,plot_group_lasso:[7,14],plot_jax_copt:[8,14],plot_nol:[7,10,11],plot_overlapping_group_lasso:[10,13],plot_pdhg:[10,12],plot_pdhg_nol:10,plot_saga_vs_svrg:[9,14],plot_sparse_nuclear_norm:[11,13],plot_steps:[2,4],plot_to:[7,10,11,12],plot_tos_nol:12,plot_tv_deblur:[12,13],plot_vertex_overlap:[3,4],plt:[0,1,2,3,6,7,8,9,10,11,12],png:[],pock:23,point:[25,26,28],poor:[1,3],possibl:[23,27,40],preprint:28,preprocess:10,prev_overlap:[1,3],previou:[1,3,16],primal:[12,23,39],print:[0,1,2,3,7,10,11,12,25,26,28],problem:[6,7,9,16,23,24,26,27,28,31],procedur:27,proceed:27,process:[16,24,25,26],product:33,program:23,progress:9,project:[17,19,20,21,36],provid:8,prox:[7,8,9,10,11,24,25,26],prox_1:[23,27,28],prox_1_arg:[],prox_2:[23,27,28],prox_factori:9,prox_tv1d_col:12,prox_tv1d_row:12,proxim:[7,23,24,25,27,39,40],proximal_gradi:[6,7],pseudo:35,pure:39,purpos:39,pylab:[0,1,2,3,6,7,8,9,10,11,12],python3:[0,1,2,3,6,7,8,9,10,11,12],python:[0,1,2,3,5,6,7,8,9,10,11,12,39],quantifi:[1,3],rand:[6,7,9,11],randint:10,randn:[6,7,9,10,11,12],random:[6,7,8,9,10,11,12],rang:[7,10,11],rangl:16,rank:[5,13,42],rate:23,ravel:[1,3,11,12],rcv1:[0,1,2,3,19],reach:[6,7],readi:40,recoveri:24,reduc:[9,28,39],refer:[9,21,23,24,25,26,27,28,33],regress:[6,9],regular:[5,9,10,11,13,14,24,42],rel:[7,10,11,12],reli:16,remi:[25,26],remov:[6,9],replac:[1,3],repres:[16,21,23,24,25,26,27,28],requir:16,res:[21,23,24,27],reshap:[11,12],resiz:12,respect:[1,3,26],result:[6,7,9,10,11,12,21,23,24,25,26,27,28],result_apgd:6,result_pgd:6,result_saga:9,result_svrg:9,return_gradi:[21,23,27],return_singular_vector:[11,12],revisit:[16,21],right:[0,1,2,3],routin:16,row:[1,3],rubinov:[16,21],run:[0,1,2,3,6,7,8,9,10,11,12],runtimewarn:[6,7],s11228:[],s_t:[1,3],saga:[5,14,25,26,28,39],same:[1,3,21],sampl:18,scalabl:[25,26],scale:[7,10,12,39],scatterpoint:[7,10,11,12],scheme:27,scipi:[1,3,7,11,12,17,19,20,21,23,24,25,26,27,28,39],script:[0,1,2,3,6,7,8,9,10,11,12],search:[7,10,11,16,27],second:[0,1,2,3,6,7,8,9,10,11,12],see:[21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],seed:[6,7,9,10,11,12],select:[1,3,16],self:[29,30,31,32,33,34,35,36,37],set:[0,2,10,16,27],set_titl:[1,3,7,10,11,12],set_xlabel:[1,3,7,10,11,12],set_xlim:[10,11,12],set_xtick:[7,10,11,12],set_ylabel:[1,3,7,10,11,12],set_ylim:[7,10,11,12],set_yscal:[7,10,11,12],set_ytick:[7,10,11,12],shape:[0,1,2,3,8,11,12,35,36,37],sharei:[7,10,11,12],should:[16,21,23,33],show:[0,1,2,3,6,7,8,9,10,11,12],sigma:[7,11,33],sigma_2:11,sigma_hat:11,sigmoid:33,sign:10,signal:24,signatur:[16,29,30,31,32,33,34,35,36,37],similar:39,simon:[16,25,26],simpl:16,simplest:16,sinc:[1,3,16],singl:[1,3,23,24,27],singular:[36,37],site:[0,1,2,3,6,7,8,9,10,11,12],size:[1,3,4,5,7,10,11,12,16,21,24,25,26,27,28],sklearn:[8,10],slightli:[0,1,2,3],smooth:[23,27,40],snapshot:9,sol:8,solut:[16,21,23,24,25,26,27,28],solv:[9,16,24,25,26,27,28,31],solver:[7,10,11,12],some:[7,10,11,25,26,28],sometim:[9,16],sourc:[0,1,2,3,5,6,7,8,9,10,11,12,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],spars:[1,3,5,7,9,12,13,16,17,19,20,21,23,26,42],specifi:16,speed:[0,2,6],sphinx:[0,1,2,3,5,6,7,8,9,10,11,12],splinalg:[1,3,11,12],split:[10,12,23,27,28,39],sqrt:10,squar:[8,10,34],squareloss:[7,12],ss_0:[],standardscal:10,start:[25,26,27,28],startswith:[1,3],state:39,step:[1,3,4,5,7,10,11,12,16,21,24,25,26,27,28],step_siz:[0,1,2,3,6,7,9,10,11,12,16,21,22,23,24,25,26,27,28],step_size2:[10,12,23],stochast:[9,25,26,39],stop:[21,25,26,27,28],store:[0,1,2,3],strategi:[0,2,16,24],strongli:26,suboptim:[6,9],subplot:[1,3,7,10,11,12],subplots_adjust:[7,10,11,12],subset:19,success:[21,23,24,25,26,27,28],successfulli:[21,23,24,25,26,27,28],sum:[8,9,12,31,32,36,37],sum_:[25,26,28,33],sum_i:[16,32],support:26,svd:[11,12],svrg:[5,14,39],symptomat:[1,3],synthet:11,system:[16,25,26],take:[16,17,19,20,21,23,24,27],teboul:24,tensorflow:8,term:[23,40],termin:[21,23,24,25,26,27,28],than:[16,26,28,31],thank:[],theori:23,thi:[1,3,8,9,10,16,17,19,20,22,23,25,26,27,28,29,31],thoma:23,thread:[26,28],three:[10,12,27,28,39],threshold:11,through:[7,16,25,26,28,40],tight_layout:[0,1,2,3],time:[0,1,2,3,6,7,8,9,10,11,12],titl:[0,2,6,9],tmp1:12,tmp2:12,todo:[],toi:16,tol:[1,3,6,7,9,10,11,12,21,22,23,24,25,26,27,28,35],toler:[6,7,10,11,12,21,25,26,27,28],tos:[10,11],tos_l:[10,11],total:[0,1,2,3,4,5,6,7,8,9,10,11,13,14,35,42],tpu:8,trace:[0,1,2,3,6,7,8,9,10,11,12,25,26,28,36,37],trace_func:[25,26,28],trace_fx:[6,8,9],trace_l:[7,10,11,12],trace_nol:[7,10,11,12],trace_pdhg:[10,12],trace_pdhg_nol:10,trace_tim:[10,11,12,25,26,28],trace_x:[7,10,11,12],tracenorm:11,track:9,train:8,triangl:16,truth:7,tv_prox:12,twice:[1,3],two:[1,3,9],type:[21,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],typic:16,unlik:16,unsur:16,updat:[4,5,16,21],url:20,usag:[0,1,2,3,6,7,8,9,10,11,12],use:[1,3,8,16,21,24,26,28,40],used:[8,16,21],useful:[25,26,28],user:16,userwarn:[0,1,2,3,6,7,8,9,10,11,12],using:[0,1,2,3,6,7,8,9,10,11,12,16,39],usr:[0,1,2,3,6,7,8,9,10,11,12],util:[0,1,2,3,6,7,8,9,10,11,12],valu:[7,8,10,11,12,16,17,19,20,21,23,24,27,31,32,33,36,37],value_and_grad:8,varianc:[9,28,39],variant:[1,3,16,21,24,25],variants_fw:[0,2],variat:[5,13,27,35,42],vector:[1,3,7,16,21,33],verbos:[1,3,7,10,11,12,21,22,23,24,25,26,27,28],veri:16,verifi:33,version:[17,19,20],vertex:[1,3,16],vrto:28,wai:39,warn:8,well:16,when:[21,23,33],whenev:[21,25,26,28],where:[16,23,24,25,27,33,34],whether:[19,20,24,25,26,27,28],which:[0,1,2,3,6,7,8,9,10,11,12,16,21,23,24,25,26,27,28],why:16,within:8,without:[7,10,11],wolf:[4,21,39],work:39,worst:16,wotao:27,written:39,www:[17,19,20],x_i:32,x_mat:12,xla_bridg:8,xlabel:[0,2,6,8,9],xlim:[6,7,9,10,11,12],xx_0:[],xx_1:[],xx_t:[],xxx:[],yin:27,ylabel:[0,2,6,8,9],ylim:[6,9],ymin:[6,9],you:39,your:40,yscale:[0,2,6,8,9],zero:[0,1,2,3,6,7,8,9,10,11,12],zip:[1,3,5]},titles:["Comparison of different step-sizes in Frank-Wolfe","Update Direction Overlap in Frank-Wolfe","Comparison of different step-sizes in Frank-Wolfe","Update Direction Overlap in Frank-Wolfe","Computation times","Examples","Accelerated gradient descent","Group Lasso regularization","Combining COPT with JAX","SAGA vs SVRG","Group lasso with overlap","Estimating a sparse and low rank matrix","Total variation regularization","Computation times","Computation times","Datasets","Frank-Wolfe and other projection-free algorithms","copt.datasets.load_covtype","copt.datasets.load_img1","copt.datasets.load_rcv1","copt.datasets.load_url","copt.minimize_frank_wolfe","copt.minimize_pairwise_frank_wolfe","copt.minimize_primal_dual","copt.minimize_proximal_gradient","copt.minimize_saga","copt.minimize_svrg","copt.minimize_three_split","copt.minimize_vrtos","copt.utils.GroupL1","copt.utils.HuberLoss","copt.utils.L1Ball","copt.utils.L1Norm","copt.utils.LogLoss","copt.utils.SquareLoss","copt.utils.TotalVariation2D","copt.utils.TraceBall","copt.utils.TraceNorm","Incremental methods","Welcome to copt!","Loss and regularization functions","Gradient-based methods","Proximal splitting"],titleterms:{"function":40,acceler:6,algorithm:[16,39],api:[],base:41,combin:8,comparison:[0,2],comput:[4,13,14],copt:[8,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,39],cucu:[],dataset:[15,17,18,19,20],descent:6,differ:[0,2],direct:[1,3],estim:11,exampl:[5,16,42],frank:[0,1,2,3,5,16],free:16,get:39,gradient:[6,41],group:[7,10],groupl1:29,huberloss:30,increment:38,jax:8,l1ball:31,l1norm:32,lasso:[7,10],load_covtyp:17,load_img1:18,load_rcv1:19,load_url:20,logloss:33,loss:40,low:11,matrix:11,method:[38,41],minimize_frank_wolf:21,minimize_pairwise_frank_wolf:22,minimize_primal_du:23,minimize_proximal_gradi:24,minimize_saga:25,minimize_svrg:26,minimize_three_split:27,minimize_vrto:28,optim:39,other:16,overlap:[1,3,10],pairwis:16,philosophi:39,project:16,proxim:[5,42],rank:11,refer:[10,16],regular:[7,12,40],saga:9,size:[0,2],spars:11,split:[5,42],squareloss:34,start:39,step:[0,2],svrg:9,time:[4,13,14],total:12,totalvariation2d:35,tracebal:36,tracenorm:37,updat:[1,3],util:[29,30,31,32,33,34,35,36,37],variat:12,welcom:39,wolf:[0,1,2,3,5,16]}})